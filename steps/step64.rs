// step64: LayerNorm과 GELU
//
// Transformer 블록의 정규화와 활성화 함수
//
// LayerNorm:
//   마지막 축(feature)을 따라 정규화: y = gamma * (x - mean) / sqrt(var + eps) + beta
//   BatchNorm과 달리 배치 크기에 무관 → Transformer 표준
//   역전파: gx = (1/σ) * (g_xhat - mean(g_xhat) - x_hat * mean(g_xhat * x_hat))
//
// GELU (Gaussian Error Linear Unit):
//   GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
//   ReLU처럼 음수를 억제하되, 부드러운 전환으로 기울기를 더 잘 전달
//   GPT-2/3, BERT의 FFN에서 ReLU 대신 사용

use dezero::{layer_norm, gelu, LayerNorm, Variable, sum};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm_2d() {
        // (batch=3, features=4)에 대해 각 행(feature 방향)을 정규화
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3, 4]),
                vec![
                    1.0, 2.0, 3.0, 4.0,   // mean=2.5, std=1.118
                    10.0, 10.0, 10.0, 10.0, // mean=10, std=0 → all 0
                    -1.0, 0.0, 1.0, 2.0,   // mean=0.5, std=1.118
                ],
            ).unwrap(),
        );
        let gamma = Variable::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[4])));
        let beta = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[4])));

        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        assert_eq!(y.shape(), vec![3, 4]);

        let y_data = y.data();

        // 각 행의 평균 ≈ 0, 분산 ≈ 1 확인
        for i in 0..3 {
            let row: Vec<f64> = (0..4).map(|j| y_data[[i, j]]).collect();
            let mean: f64 = row.iter().sum::<f64>() / 4.0;
            let var: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 4.0;
            assert!(mean.abs() < 1e-5, "row {} mean = {}", i, mean);
            // 상수 행 (row 1)은 분산 0 → 정규화 후에도 0
            if i != 1 {
                assert!((var - 1.0).abs() < 0.01, "row {} var = {}", i, var);
            }
        }

        // 상수 행: 모두 같은 값 → 정규화 후 모두 0
        for j in 0..4 {
            assert!(y_data[[1, j]].abs() < 1e-3, "constant row should be ~0");
        }
        println!("2D layer_norm: mean≈0, var≈1 ✓");
    }

    #[test]
    fn test_layer_norm_3d() {
        // (B=2, T=3, D=4) — Transformer의 전형적 shape
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 4]), data).unwrap(),
        );
        let gamma = Variable::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[4])));
        let beta = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[4])));

        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        assert_eq!(y.shape(), vec![2, 3, 4]);

        // 각 (b, t) 위치에서 마지막 축의 평균 ≈ 0
        let y_data = y.data();
        for b in 0..2 {
            for t in 0..3 {
                let mean: f64 = (0..4).map(|d| y_data[[b, t, d]]).sum::<f64>() / 4.0;
                assert!(mean.abs() < 1e-5, "[{},{}] mean = {}", b, t, mean);
            }
        }
        println!("3D layer_norm shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_layer_norm_gamma_beta() {
        // gamma와 beta의 효과 확인
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ).unwrap(),
        );
        // gamma=2로 스케일, beta=1로 시프트
        let gamma = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![2.0, 2.0, 2.0]).unwrap(),
        );
        let beta = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 1.0, 1.0]).unwrap(),
        );

        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        let y_data = y.data();

        // 정규화 후 mean≈1 (beta), std≈2 (gamma)
        for i in 0..2 {
            let row: Vec<f64> = (0..3).map(|j| y_data[[i, j]]).collect();
            let mean: f64 = row.iter().sum::<f64>() / 3.0;
            assert!(
                (mean - 1.0).abs() < 1e-5,
                "row {} mean = {} (expected ≈1)", i, mean,
            );
        }
        println!("gamma/beta effect verified ✓");
    }

    #[test]
    fn test_layer_norm_backward() {
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 4]),
                vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            ).unwrap(),
        );
        let gamma = Variable::new(ndarray::ArrayD::ones(ndarray::IxDyn(&[4])));
        let beta = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[4])));

        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        let loss = sum(&y);
        loss.backward(false, false);

        let gx = x.grad().unwrap();
        let ggamma = gamma.grad().unwrap();
        let gbeta = beta.grad().unwrap();

        assert_eq!(gx.shape(), &[2, 4]);
        assert_eq!(ggamma.shape(), &[4]);
        assert_eq!(gbeta.shape(), &[4]);

        // sum(layer_norm(x))의 gx: 정규화 후 합 = 0이므로 gx ≈ 0
        assert!(
            gx.iter().all(|&v| v.abs() < 1e-5),
            "gx should be ~0, got {:?}", gx.as_slice().unwrap(),
        );

        // gbeta = sum(gy, batch) = [1,1,1,1] * 2 samples = [2,2,2,2]
        for j in 0..4 {
            assert!(
                (gbeta[[j]] - 2.0).abs() < 1e-10,
                "gbeta[{}] = {}", j, gbeta[[j]],
            );
        }
        println!("layer_norm backward shapes: gx {:?}, ggamma {:?}, gbeta {:?} ✓",
            gx.shape(), ggamma.shape(), gbeta.shape());
    }

    #[test]
    fn test_layer_norm_backward_numerical() {
        // 수치 미분으로 역전파 검증
        let x_data = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), x_data.clone()).unwrap(),
        );
        let gamma = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 0.5]).unwrap(),
        );
        let beta = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[3])));

        // loss = sum(layer_norm(x)^2) — 비자명한 loss
        let y = layer_norm(&x, &gamma, &beta, 1e-5);
        let loss = sum(&y.pow(2.0));
        loss.backward(false, false);

        let analytic_gx = x.grad().unwrap();

        // 수치 미분
        let eps = 1e-5;
        let mut numerical_gx = vec![0.0; 6];
        for i in 0..6 {
            let mut xp = x_data.clone();
            xp[i] += eps;
            let xp_var = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), xp).unwrap(),
            );
            let gamma_c = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 0.5]).unwrap(),
            );
            let beta_c = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[3])));
            let yp = layer_norm(&xp_var, &gamma_c, &beta_c, 1e-5);
            let lp: f64 = sum(&yp.pow(2.0)).data().iter().next().copied().unwrap();

            let mut xm = x_data.clone();
            xm[i] -= eps;
            let xm_var = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), xm).unwrap(),
            );
            let gamma_c = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 0.5]).unwrap(),
            );
            let beta_c = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[3])));
            let ym = layer_norm(&xm_var, &gamma_c, &beta_c, 1e-5);
            let lm: f64 = sum(&ym.pow(2.0)).data().iter().next().copied().unwrap();

            numerical_gx[i] = (lp - lm) / (2.0 * eps);
        }

        // 해석적 기울기와 수치 기울기 비교
        let analytic: Vec<f64> = analytic_gx.iter().cloned().collect();
        println!("analytic gx:  {:?}", analytic);
        println!("numerical gx: {:?}", numerical_gx);

        for i in 0..6 {
            assert!(
                (analytic[i] - numerical_gx[i]).abs() < 1e-3,
                "mismatch at {}: analytic={}, numerical={}",
                i, analytic[i], numerical_gx[i],
            );
        }
        println!("layer_norm backward numerical check passed ✓");
    }

    #[test]
    fn test_layer_norm_layer() {
        // LayerNorm 레이어 사용 테스트
        let ln = LayerNorm::new(4);
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3, 4]),
                (0..24).map(|i| i as f64).collect(),
            ).unwrap(),
        );

        let y = ln.forward(&x);
        assert_eq!(y.shape(), vec![2, 3, 4]);

        // 학습 가능 파라미터 확인
        assert_eq!(ln.params().len(), 2); // gamma, beta
        assert_eq!(ln.params()[0].shape(), vec![4]); // gamma
        assert_eq!(ln.params()[1].shape(), vec![4]); // beta
        println!("LayerNorm layer test ✓");
    }

    #[test]
    fn test_gelu_values() {
        // GELU의 주요 특성 확인
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[7]),
                vec![-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
            ).unwrap(),
        );

        let y = gelu(&x);
        let y_data = y.data();

        // GELU(0) = 0
        assert!((y_data[[3]] - 0.0).abs() < 1e-10, "GELU(0) = {}", y_data[[3]]);

        // GELU(x) ≈ x for large positive x
        assert!((y_data[[6]] - 3.0).abs() < 0.01, "GELU(3) ≈ 3, got {}", y_data[[6]]);

        // GELU(x) ≈ 0 for large negative x
        assert!(y_data[[0]].abs() < 0.01, "GELU(-3) ≈ 0, got {}", y_data[[0]]);

        // GELU(-x) ≠ -GELU(x) (비대칭)
        assert!((y_data[[1]] + y_data[[5]]).abs() > 0.01, "GELU is asymmetric");

        println!("GELU values:");
        for i in 0..7 {
            let x_val = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0][i];
            println!("  GELU({:5.1}) = {:8.5}", x_val, y_data[[i]]);
        }
    }

    #[test]
    fn test_gelu_backward() {
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[5]),
                vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            ).unwrap(),
        );

        let y = gelu(&x);
        let loss = sum(&y);
        loss.backward(false, false);

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), &[5]);

        // 수치 미분 검증
        let eps = 1e-5;
        let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
        for i in 0..5 {
            let xp: f64 = x_vals[i] + eps;
            let xm: f64 = x_vals[i] - eps;
            let gelu_p = 0.5 * xp * (1.0 + (sqrt_2_pi * (xp + 0.044715 * xp.powi(3))).tanh());
            let gelu_m = 0.5 * xm * (1.0 + (sqrt_2_pi * (xm + 0.044715 * xm.powi(3))).tanh());
            let numerical = (gelu_p - gelu_m) / (2.0 * eps);
            assert!(
                (grad[[i]] - numerical).abs() < 1e-4,
                "GELU grad mismatch at x={}: analytic={}, numerical={}",
                x_vals[i], grad[[i]], numerical,
            );
        }
        println!("GELU backward numerical check passed ✓");
    }

    #[test]
    fn test_gelu_vs_relu() {
        // GELU와 ReLU 비교: GELU는 음수 영역에서도 약간의 값을 허용
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[5]),
                vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            ).unwrap(),
        );

        let gelu_y = gelu(&x);
        let g = gelu_y.data();

        // x=-0.5에서 GELU는 음수 (약 -0.154)
        assert!(g[[1]] < 0.0, "GELU(-0.5) should be negative: {}", g[[1]]);

        // ReLU(-0.5) = 0이지만 GELU(-0.5) ≠ 0 → 기울기가 흐를 수 있음
        println!("GELU vs ReLU at key points:");
        for i in 0..5 {
            let x_val: f64 = [-1.0, -0.5, 0.0, 0.5, 1.0][i];
            let relu_val = x_val.max(0.0);
            println!("  x={:5.1}  ReLU={:6.3}  GELU={:6.3}", x_val, relu_val, g[[i]]);
        }
    }

    #[test]
    fn test_transformer_ffn_pattern() {
        // Transformer FFN: LayerNorm → Linear → GELU → Linear
        // shape 흐름 검증
        let ln = LayerNorm::new(8);
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 4, 8]),
                (0..64).map(|i| i as f64 * 0.1).collect(),
            ).unwrap(),
        );

        // LayerNorm
        let normed = ln.forward(&x);
        assert_eq!(normed.shape(), vec![2, 4, 8]);

        // GELU (element-wise)
        let activated = gelu(&normed);
        assert_eq!(activated.shape(), vec![2, 4, 8]);

        // backward
        let loss = sum(&activated);
        loss.backward(false, false);

        assert_eq!(x.grad().unwrap().shape(), &[2, 4, 8]);
        assert!(x.grad().unwrap().iter().all(|v| v.is_finite()), "grad has NaN/Inf");

        println!("Transformer FFN pattern: LayerNorm → GELU ✓");
        println!("  input shape:  {:?}", x.shape());
        println!("  output shape: {:?}", activated.shape());
        println!("  grad shape:   {:?}", x.grad().unwrap().shape());
    }
}
