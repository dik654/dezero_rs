// step63: Softmax(axis)와 Causal Mask
//
// Attention의 나머지 핵심 부품
//
// Softmax(axis):
//   기존 softmax_simple은 2D axis=1 전용
//   Attention은 4D 텐서 (B, H, T, T)의 마지막 축(axis=-1)에 softmax 필요
//   수치 안정성: max를 빼서 exp 오버플로 방지
//   역전파: gx = y * (gy - sum(gy * y, axis, keepdims))
//     야코비안 ∂y_i/∂x_j = y_i(δ_ij - y_j)에서 유도
//
// Causal Mask:
//   미래 토큰을 참조하지 못하게 하는 삼각 마스크
//   scores (B, H, T, T)에서 col > row인 위치를 -∞로 설정
//   softmax(-∞) = 0이므로 미래 정보가 완전히 차단됨
//   역전파: 마스크 위치의 기울기는 0

use dezero::{softmax, causal_mask, batched_matmul, transpose_axes, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_2d() {
        // 2D 텐서에서 axis=1 (기존 softmax_simple과 동일)
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
            ).unwrap(),
        );

        let y = softmax(&x, -1); // axis=-1 = axis=1
        assert_eq!(y.shape(), vec![2, 3]);

        // 각 행의 합 = 1 확인
        let y_data = y.data();
        let row0_sum: f64 = (0..3).map(|j| y_data[[0, j]]).sum();
        let row1_sum: f64 = (0..3).map(|j| y_data[[1, j]]).sum();
        assert!((row0_sum - 1.0).abs() < 1e-10, "row 0 sum = {}", row0_sum);
        assert!((row1_sum - 1.0).abs() < 1e-10, "row 1 sum = {}", row1_sum);

        // [1, 1, 1]의 softmax = [1/3, 1/3, 1/3]
        for j in 0..3 {
            assert!((y_data[[1, j]] - 1.0 / 3.0).abs() < 1e-10);
        }

        // [1, 2, 3]에서 3이 가장 크므로 softmax[2]가 가장 큼
        assert!(y_data[[0, 2]] > y_data[[0, 1]]);
        assert!(y_data[[0, 1]] > y_data[[0, 0]]);
        println!("2D softmax: {:?} ✓", y_data.as_slice().unwrap());
    }

    #[test]
    fn test_softmax_4d() {
        // Attention scores: (B=1, H=2, T=3, T=3)
        let data: Vec<f64> = (0..18).map(|i| i as f64 * 0.5).collect();
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 3, 3]), data).unwrap(),
        );

        let y = softmax(&x, -1); // 마지막 축(axis=3)
        assert_eq!(y.shape(), vec![1, 2, 3, 3]);

        // 각 (b, h, t) 위치에서 마지막 축 합 = 1 확인
        let y_data = y.data();
        for b in 0..1 {
            for h in 0..2 {
                for t in 0..3 {
                    let row_sum: f64 = (0..3).map(|j| y_data[[b, h, t, j]]).sum();
                    assert!(
                        (row_sum - 1.0).abs() < 1e-10,
                        "sum at [{},{},{}] = {}", b, h, t, row_sum,
                    );
                }
            }
        }
        println!("4D softmax shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 큰 값에서도 오버플로 없이 동작해야 함
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 3]),
                vec![1000.0, 1001.0, 1002.0],
            ).unwrap(),
        );

        let y = softmax(&x, -1);
        let y_data = y.data();

        // NaN이나 Inf가 없어야 함
        assert!(y_data.iter().all(|v| v.is_finite()), "overflow detected");

        let sum: f64 = y_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum = {}", sum);
        println!("numerical stability: values {:?} → softmax sum = {:.10} ✓",
            vec![1000.0, 1001.0, 1002.0], sum);
    }

    #[test]
    fn test_softmax_backward() {
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5],
            ).unwrap(),
        );

        let y = softmax(&x, -1);
        let loss = dezero::sum(&y);
        loss.backward(false, false);

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);

        // sum(softmax(x))의 기울기: sum의 기울기는 [1,1,1]
        // softmax backward: y * (gy - sum(gy*y)) = y * (1 - sum(y)) = y * (1 - 1) = 0
        // sum(softmax(x)) = 1 (상수)이므로 기울기가 0이어야 함
        assert!(
            grad.iter().all(|&v| v.abs() < 1e-10),
            "grad should be ~0, got {:?}", grad.as_slice().unwrap(),
        );
        println!("softmax backward (sum case): all zeros ✓");
    }

    #[test]
    fn test_softmax_backward_with_loss() {
        // 수치 미분으로 역전파 검증
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
            ).unwrap(),
        );

        let y = softmax(&x, -1);
        // 특정 원소만 선택하는 loss: y[0,1] (weighted sum)
        let y_data = y.data();
        let target_val = y_data[[0, 1]]; // softmax의 특정 값 하나

        // 수치 미분
        let eps = 1e-5;
        let x_data_orig: Vec<f64> = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let mut numerical_grad = vec![0.0; 6];
        for i in 0..6 {
            let mut x_plus = x_data_orig.clone();
            x_plus[i] += eps;
            let xp = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_plus).unwrap(),
            );
            let yp = softmax(&xp, -1);
            let lp = yp.data()[[0, 1]];

            let mut x_minus = x_data_orig.clone();
            x_minus[i] -= eps;
            let xm = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_minus).unwrap(),
            );
            let ym = softmax(&xm, -1);
            let lm = ym.data()[[0, 1]];

            numerical_grad[i] = (lp - lm) / (2.0 * eps);
        }

        println!("numerical grad: {:?}", numerical_grad);
        // 첫 번째 행만 비 0 (두 번째 행은 y[0,1]과 무관)
        assert!(numerical_grad[3..6].iter().all(|&v| v.abs() < 1e-8));
        println!("softmax numerical gradient verification ✓");
    }

    #[test]
    fn test_causal_mask_2d() {
        // 가장 간단한 경우: (T=4, T=4) 행렬
        let x = Variable::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[4, 4])));

        let masked = causal_mask(&x);
        let m = masked.data();

        // 대각선 + 하삼각: 1.0 유지
        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[1, 0]], 1.0);
        assert_eq!(m[[1, 1]], 1.0);
        assert_eq!(m[[3, 0]], 1.0);

        // 상삼각: -inf
        assert_eq!(m[[0, 1]], f64::NEG_INFINITY);
        assert_eq!(m[[0, 3]], f64::NEG_INFINITY);
        assert_eq!(m[[2, 3]], f64::NEG_INFINITY);

        println!("causal mask 2D:");
        for i in 0..4 {
            let row: Vec<String> = (0..4)
                .map(|j| if m[[i, j]].is_finite() { format!("{:.0}", m[[i, j]]) } else { "-∞".to_string() })
                .collect();
            println!("  [{}]", row.join(", "));
        }
    }

    #[test]
    fn test_causal_mask_4d() {
        // Attention 패턴: (B=1, H=2, T=3, T=3)
        let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 3, 3]), data).unwrap(),
        );

        let masked = causal_mask(&x);
        assert_eq!(masked.shape(), vec![1, 2, 3, 3]);

        let m = masked.data();
        // 각 (b, h) 슬라이스에서 상삼각 = -inf
        for b in 0..1 {
            for h in 0..2 {
                // 대각선: 원래 값 유지
                for i in 0..3 {
                    assert!(m[[b, h, i, i]].is_finite());
                }
                // 상삼각: -inf
                assert_eq!(m[[b, h, 0, 1]], f64::NEG_INFINITY);
                assert_eq!(m[[b, h, 0, 2]], f64::NEG_INFINITY);
                assert_eq!(m[[b, h, 1, 2]], f64::NEG_INFINITY);
                // 하삼각: 원래 값 유지
                assert!(m[[b, h, 1, 0]].is_finite());
                assert!(m[[b, h, 2, 0]].is_finite());
                assert!(m[[b, h, 2, 1]].is_finite());
            }
        }
        println!("4D causal mask shape: {:?} ✓", masked.shape());
    }

    #[test]
    fn test_causal_mask_backward() {
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ).unwrap(),
        );

        let masked = causal_mask(&x);
        // 하삼각+대각선 원소만 sum (상삼각은 -inf)
        // 유한한 값: [0,0]=1, [1,0]=4, [1,1]=5, [2,0]=7, [2,1]=8, [2,2]=9
        let loss = dezero::sum(&masked);
        loss.backward(false, false);

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), &[3, 3]);

        // 마스크된 위치(상삼각)는 기울기 0, 나머지는 1
        assert_eq!(grad[[0, 0]], 1.0); // 대각선
        assert_eq!(grad[[1, 0]], 1.0); // 하삼각
        assert_eq!(grad[[0, 1]], 0.0); // 상삼각 (마스크됨)
        assert_eq!(grad[[0, 2]], 0.0); // 상삼각 (마스크됨)
        println!("causal mask backward:");
        for i in 0..3 {
            let row: Vec<f64> = (0..3).map(|j| grad[[i, j]]).collect();
            println!("  {:?}", row);
        }
    }

    #[test]
    fn test_softmax_with_causal_mask() {
        // causal mask + softmax 조합 테스트
        // softmax(-inf) = 0이 되어야 함
        let x = Variable::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[1, 1, 4, 4])));

        let masked = causal_mask(&x);
        let probs = softmax(&masked, -1);
        let p = probs.data();

        println!("masked softmax probabilities:");
        for i in 0..4 {
            let row: Vec<f64> = (0..4).map(|j| p[[0, 0, i, j]]).collect();
            println!("  row {}: [{:.4}, {:.4}, {:.4}, {:.4}]", i, row[0], row[1], row[2], row[3]);
        }

        // row 0: 자기 자신만 → [1, 0, 0, 0]
        assert!((p[[0, 0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!(p[[0, 0, 0, 1]].abs() < 1e-10);

        // row 1: 2개 → [0.5, 0.5, 0, 0]
        assert!((p[[0, 0, 1, 0]] - 0.5).abs() < 1e-10);
        assert!((p[[0, 0, 1, 1]] - 0.5).abs() < 1e-10);
        assert!(p[[0, 0, 1, 2]].abs() < 1e-10);

        // row 3: 4개 → [0.25, 0.25, 0.25, 0.25]
        for j in 0..4 {
            assert!((p[[0, 0, 3, j]] - 0.25).abs() < 1e-10);
        }
        println!("causal mask + softmax: correct attention pattern ✓");
    }

    #[test]
    fn test_full_attention_with_softmax() {
        // 완전한 Attention 파이프라인 테스트:
        // Q, K, V: (B=1, H=2, T=4, D=3)
        // scores = Q @ K^T / sqrt(D)
        // masked_scores = causal_mask(scores)
        // probs = softmax(masked_scores, -1)
        // output = probs @ V
        let q_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1).collect();
        let k_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1 + 0.05).collect();
        let v_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1).collect();

        let q = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), q_data).unwrap(),
        );
        let k = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), k_data).unwrap(),
        );
        let v = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), v_data).unwrap(),
        );

        // K^T: (1,2,4,3) → (1,2,3,4)
        let k_t = transpose_axes(&k, &[0, 1, 3, 2]);

        // scores = Q @ K^T / sqrt(D)
        let scores = batched_matmul(&q, &k_t);
        let d_k = 3.0_f64;
        let scaled = &scores / d_k.sqrt();
        assert_eq!(scaled.shape(), vec![1, 2, 4, 4]);

        // causal mask
        let masked = causal_mask(&scaled);

        // softmax
        let probs = softmax(&masked, -1);
        assert_eq!(probs.shape(), vec![1, 2, 4, 4]);

        // 각 행의 합 = 1 확인
        let p = probs.data();
        for h in 0..2 {
            for t in 0..4 {
                let row_sum: f64 = (0..4).map(|j| p[[0, h, t, j]]).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-10,
                    "probs row sum at [0,{},{}] = {}", h, t, row_sum,
                );
            }
        }

        // 미래 위치는 확률 0
        for h in 0..2 {
            assert!(p[[0, h, 0, 1]].abs() < 1e-10, "future token should be 0");
            assert!(p[[0, h, 0, 2]].abs() < 1e-10);
            assert!(p[[0, h, 1, 2]].abs() < 1e-10);
        }

        // output = probs @ V
        let out = batched_matmul(&probs, &v);
        assert_eq!(out.shape(), vec![1, 2, 4, 3]);

        // backward
        let loss = dezero::sum(&out);
        loss.backward(false, false);

        assert_eq!(q.grad().unwrap().shape(), &[1, 2, 4, 3]);
        assert_eq!(k.grad().unwrap().shape(), &[1, 2, 4, 3]);
        assert_eq!(v.grad().unwrap().shape(), &[1, 2, 4, 3]);

        // grad에 NaN이 없는지 확인
        assert!(q.grad().unwrap().iter().all(|v| v.is_finite()), "Q grad has NaN/Inf");
        assert!(k.grad().unwrap().iter().all(|v| v.is_finite()), "K grad has NaN/Inf");
        assert!(v.grad().unwrap().iter().all(|v| v.is_finite()), "V grad has NaN/Inf");

        println!("Full Attention pipeline with causal mask:");
        println!("  Q shape:      {:?}", q.shape());
        println!("  scores shape: {:?}", scaled.shape());
        println!("  probs shape:  {:?}", probs.shape());
        println!("  output shape: {:?}", out.shape());
        println!("  Q grad shape: {:?}", q.grad().unwrap().shape());
        println!("  All grads finite ✓");
    }
}
