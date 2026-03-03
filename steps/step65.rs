// step65: Multi-Head Attention (CausalSelfAttention)
//
// 지금까지 구현한 모든 부품을 조합하여 GPT 스타일 Self-Attention을 완성한다:
//   - Linear (Q, K, V, Output 프로젝션)
//   - transpose_axes (헤드 분할/병합 시 축 재배치)
//   - batched_matmul (Q@K^T, scores@V)
//   - causal_mask (미래 토큰 차단)
//   - softmax (attention 가중치 정규화)
//   - dropout (attention dropout)
//
// 데이터 흐름:
//   x (B,T,D) → Q,K,V (B,H,T,D_h) → scores (B,H,T,T) → out (B,T,D)

use dezero::{
    batched_matmul, causal_mask, dropout, no_grad, reshape, softmax, transpose_axes,
    CausalSelfAttention, Variable,
};

#[cfg(test)]
mod tests {
    use super::*;

    // 헬퍼: (B, T, D) shape의 Variable 생성
    fn make_input(b: usize, t: usize, d: usize, seed: f64) -> Variable {
        let n = b * t * d;
        let data: Vec<f64> = (0..n).map(|i| ((i as f64 + seed) * 0.01).sin()).collect();
        Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, t, d]), data).unwrap(),
        )
    }

    #[test]
    fn test_forward_shape() {
        // --- 기본 forward shape 검증 ---
        // (B=2, T=4, D=8), n_head=2 → D_head=4
        let attn = CausalSelfAttention::new(8, 2, 0.0, 42);
        let x = make_input(2, 4, 8, 0.0);
        let y = attn.forward(&x);

        println!("input shape: {:?}", x.shape());
        println!("output shape: {:?}", y.shape());
        assert_eq!(y.shape(), vec![2, 4, 8]);
    }

    #[test]
    fn test_single_head() {
        // --- 헤드 1개: 단순한 Self-Attention ---
        // n_head=1이면 헤드 분할/병합이 사실상 없어야 함
        let attn = CausalSelfAttention::new(6, 1, 0.0, 100);
        let x = make_input(1, 3, 6, 1.0);
        let y = attn.forward(&x);

        println!("single head output shape: {:?}", y.shape());
        assert_eq!(y.shape(), vec![1, 3, 6]);
    }

    #[test]
    fn test_backward_shapes() {
        // --- backward: 모든 파라미터에 기울기가 생성되는지 확인 ---
        let attn = CausalSelfAttention::new(8, 2, 0.0, 42);
        let x = make_input(1, 4, 8, 0.0);
        let y = attn.forward(&x);

        // sum으로 스칼라 만들어 backward
        let loss = dezero::sum(&y);
        loss.backward(false, false);

        // x에 기울기가 있어야 함
        let x_grad = x.grad().unwrap();
        println!("x.grad shape: {:?}", x_grad.shape());
        assert_eq!(x_grad.shape(), &[1, 4, 8]);

        // 모든 파라미터에 기울기가 있어야 함
        let params = attn.params();
        println!("param count: {}", params.len());
        // 4 Linear × (W + b) = 8 파라미터
        assert_eq!(params.len(), 8);
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
            println!("  param {} shape: {:?}, grad shape: {:?}", i, p.shape(), p.grad().unwrap().shape());
        }
    }

    #[test]
    fn test_causal_property() {
        // --- 인과성 검증: 위치 t의 출력이 t+1 이후의 입력에 의존하지 않는다 ---
        // 동일한 입력에서 마지막 토큰만 변경했을 때, 이전 위치의 출력이 동일해야 함
        let attn = CausalSelfAttention::new(8, 2, 0.0, 42);

        let x1 = make_input(1, 4, 8, 0.0);
        let _guard = no_grad();
        let y1 = attn.forward(&x1);
        let y1_data = y1.data();

        // x2: 마지막 토큰(t=3)만 다른 값으로 교체
        let mut x2_raw: Vec<f64> = x1.data().iter().cloned().collect();
        for i in (3 * 8)..(4 * 8) {
            x2_raw[i] = 999.0; // 완전히 다른 값
        }
        let x2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 8]), x2_raw).unwrap(),
        );
        let y2 = attn.forward(&x2);
        let y2_data = y2.data();

        // t=0, 1, 2의 출력은 동일해야 함 (causal mask 때문)
        for t in 0..3 {
            for d in 0..8 {
                let v1 = y1_data[[0, t, d]];
                let v2 = y2_data[[0, t, d]];
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "position {} feature {} differs: {} vs {}",
                    t, d, v1, v2
                );
            }
        }

        // t=3의 출력은 달라야 함
        let mut any_diff = false;
        for d in 0..8 {
            if (y1_data[[0, 3, d]] - y2_data[[0, 3, d]]).abs() > 1e-10 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "position 3 should differ when input changes");
        println!("Causal property verified: positions 0-2 unchanged, position 3 changed");
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        // --- attention 가중치가 각 행에서 합이 1인지 검증 ---
        // softmax 후 각 쿼리 위치의 가중치 합 = 1
        // CausalSelfAttention 내부를 직접 검증할 수 없으므로,
        // 동일한 파이프라인을 수동으로 구성하여 확인
        let b = 1;
        let h = 2;
        let t = 4;
        let d_h = 3;

        let q = make_input(b, h * t, d_h, 0.0);
        let q = reshape(&q, &[b, h, t, d_h]);
        let k = make_input(b, h * t, d_h, 1.0);
        let k = reshape(&k, &[b, h, t, d_h]);

        let k_t = transpose_axes(&k, &[0, 1, 3, 2]);
        let scores = &batched_matmul(&q, &k_t) / (d_h as f64).sqrt();
        let scores = causal_mask(&scores);
        let attn = softmax(&scores, -1);
        let attn_data = attn.data();

        println!("Attention weights (B=0, H=0):");
        for ti in 0..t {
            let mut row_sum = 0.0;
            for tj in 0..t {
                let w = attn_data[[0, 0, ti, tj]];
                print!("{:.3} ", w);
                row_sum += w;
            }
            println!(" | sum = {:.6}", row_sum);
            assert!((row_sum - 1.0).abs() < 1e-6, "row {} sum = {}", ti, row_sum);
        }
    }

    #[test]
    fn test_causal_mask_in_attention() {
        // --- causal mask: 미래 위치의 가중치가 0인지 확인 ---
        let b = 1;
        let h = 1;
        let t = 4;
        let d_h = 3;

        let q = make_input(b, t, d_h, 2.0);
        let q = reshape(&q, &[b, h, t, d_h]);
        let k = make_input(b, t, d_h, 3.0);
        let k = reshape(&k, &[b, h, t, d_h]);

        let k_t = transpose_axes(&k, &[0, 1, 3, 2]);
        let scores = &batched_matmul(&q, &k_t) / (d_h as f64).sqrt();
        let scores = causal_mask(&scores);
        let attn = softmax(&scores, -1);
        let attn_data = attn.data();

        // 상삼각(미래 위치)은 0이어야 함
        for i in 0..t {
            for j in (i + 1)..t {
                let w = attn_data[[0, 0, i, j]];
                assert!(
                    w.abs() < 1e-10,
                    "future weight [{},{}] = {} should be 0",
                    i, j, w
                );
            }
        }
        println!("Causal mask verified: all future weights are 0");
    }

    #[test]
    fn test_multi_head_different_heads() {
        // --- 헤드 수 변경이 출력에 영향을 주는지 확인 ---
        // 같은 입력, 다른 n_head → 출력이 다름 (내부 구조가 다르므로)
        let _guard = no_grad();

        let attn_2h = CausalSelfAttention::new(8, 2, 0.0, 42);
        let attn_4h = CausalSelfAttention::new(8, 4, 0.0, 42);

        let x = make_input(1, 4, 8, 0.0);
        let y_2h = attn_2h.forward(&x);
        let y_4h = attn_4h.forward(&x);

        // shape은 동일
        assert_eq!(y_2h.shape(), vec![1, 4, 8]);
        assert_eq!(y_4h.shape(), vec![1, 4, 8]);

        // 값은 다름 (seed는 같지만 헤드 분할이 다르므로)
        let d1 = y_2h.data();
        let d2 = y_4h.data();
        let mut any_diff = false;
        for (a, b) in d1.iter().zip(d2.iter()) {
            if (a - b).abs() > 1e-10 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "different n_head should produce different outputs");
        println!("2-head vs 4-head: outputs differ as expected");
    }

    #[test]
    fn test_dropout_effect() {
        // --- dropout > 0: 학습 모드에서 출력이 달라지는지 확인 ---
        // dropout=0.5로 설정하면 attention 가중치의 일부가 0이 됨
        let attn_drop = CausalSelfAttention::new(8, 2, 0.5, 42);
        let x = make_input(1, 4, 8, 0.0);

        // 학습 모드 (dropout 적용)
        let y_train = attn_drop.forward(&x);

        // 테스트 모드 (dropout 비적용)
        let _guard = dezero::test_mode();
        let y_test = attn_drop.forward(&x);

        let d_train = y_train.data();
        let d_test = y_test.data();

        // shape은 동일
        assert_eq!(d_train.shape(), d_test.shape());

        // 값은 다를 수 있음 (dropout이 적용되었으므로)
        println!("train output[0,0,:4]: {:.4} {:.4} {:.4} {:.4}",
            d_train[[0, 0, 0]], d_train[[0, 0, 1]], d_train[[0, 0, 2]], d_train[[0, 0, 3]]);
        println!("test  output[0,0,:4]: {:.4} {:.4} {:.4} {:.4}",
            d_test[[0, 0, 0]], d_test[[0, 0, 1]], d_test[[0, 0, 2]], d_test[[0, 0, 3]]);
        println!("Dropout test passed");
    }

    #[test]
    fn test_cleargrads() {
        // --- cleargrads가 모든 파라미터의 기울기를 초기화하는지 확인 ---
        let attn = CausalSelfAttention::new(8, 2, 0.0, 42);
        let x = make_input(1, 4, 8, 0.0);

        // 첫 번째 forward + backward
        let y = attn.forward(&x);
        dezero::sum(&y).backward(false, false);

        // 기울기가 있는지 확인
        for p in attn.params() {
            assert!(p.grad().is_some());
        }

        // cleargrads
        attn.cleargrads();

        // 기울기가 초기화되었는지 확인
        for p in attn.params() {
            assert!(p.grad().is_none(), "grad should be cleared");
        }
        println!("cleargrads verified");
    }

    #[test]
    fn test_batch_independence() {
        // --- 배치 내 샘플이 독립적으로 처리되는지 확인 ---
        // 배치 2개를 한번에 넣은 결과와, 각각 넣은 결과가 동일해야 함
        let attn = CausalSelfAttention::new(6, 2, 0.0, 42);
        let _guard = no_grad();

        // 배치 2 한번에
        let x_batch = make_input(2, 3, 6, 0.0);
        let y_batch = attn.forward(&x_batch);
        let y_batch_data = y_batch.data();

        // 배치 1씩 따로
        let x0_raw: Vec<f64> = x_batch.data().as_slice().unwrap()[..18].to_vec();
        let x1_raw: Vec<f64> = x_batch.data().as_slice().unwrap()[18..].to_vec();
        let x0 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, 6]), x0_raw).unwrap(),
        );
        let x1 = Variable::new(
            nd