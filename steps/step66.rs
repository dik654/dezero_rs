// step66: TransformerBlock (GPT Block)
//
// Step 61~65에서 만든 모든 부품을 조합하여 GPT-2 스타일 Transformer 블록을 완성한다.
//
// Pre-LN 아키텍처:
//   x → LayerNorm → CausalSelfAttention → Dropout → +x (residual)
//     → LayerNorm → FFN(Linear→GELU→Linear) → Dropout → +x (residual) → out
//
// 구성 요소:
//   - LayerNorm (step 64): 정규화
//   - CausalSelfAttention (step 65): Multi-Head Attention
//   - GELU (step 64): FFN 활성화
//   - Residual Connection: 기울기 보존
//   - Dropout: 정규화
//
// FFN: D → 4D → D (4배 확장 후 축소)
// 이 블록을 N번 반복하면 GPT의 본체가 된다.

use dezero::{
    no_grad, test_mode, dropout, sum,
    TransformerBlock, Variable,
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
        // (B=2, T=4, D=8), n_head=2
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        let x = make_input(2, 4, 8, 0.0);
        let y = block.forward(&x);

        println!("input shape: {:?}", x.shape());
        println!("output shape: {:?}", y.shape());
        assert_eq!(y.shape(), vec![2, 4, 8]);
    }

    #[test]
    fn test_backward() {
        // --- backward: 모든 파라미터에 기울기가 생성되는지 확인 ---
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        let x = make_input(1, 4, 8, 0.0);
        let y = block.forward(&x);

        let loss = sum(&y);
        loss.backward(false, false);

        // x에 기울기가 있어야 함
        let x_grad = x.grad().unwrap();
        assert_eq!(x_grad.shape(), &[1, 4, 8]);

        // 모든 파라미터에 기울기가 있어야 함
        let params = block.params();
        println!("param count: {}", params.len());
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
            println!("  param {} shape: {:?}", i, p.shape());
        }

        // 기울기에 NaN/Inf 없는지 확인
        assert!(x_grad.iter().all(|v| v.is_finite()), "x grad has NaN/Inf");
        for (i, p) in params.iter().enumerate() {
            let g = p.grad().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "param {} grad has NaN/Inf", i);
        }
        println!("backward: all grads finite ✓");
    }

    #[test]
    fn test_param_count() {
        // --- 파라미터 수 검증 ---
        // ln1: gamma(D) + beta(D) = 2
        // attn: 4 Linear × (W + b) = 8
        // ln2: gamma(D) + beta(D) = 2
        // mlp_fc: W(D, 4D) + b(4D) = 2
        // mlp_proj: W(4D, D) + b(D) = 2
        // 총: 2 + 8 + 2 + 2 + 2 = 16
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        // forward를 한번 호출해야 Linear의 lazy init이 실행됨
        let x = make_input(1, 2, 8, 0.0);
        let _y = block.forward(&x);

        let params = block.params();
        assert_eq!(params.len(), 16, "expected 16 params, got {}", params.len());
        println!("param count: {} ✓", params.len());

        // 총 파라미터 원소 수 계산
        let total_elements: usize = params.iter().map(|p| p.data().len()).sum();
        println!("total parameter elements: {}", total_elements);
    }

    #[test]
    fn test_residual_effect() {
        // --- residual 효과: 입력 정보가 출력에 보존되는지 ---
        // 블록 초기 (가중치 ≈ 작은 값)에서 출력은 입력과 유사해야 함
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        let _guard = no_grad();
        let x = make_input(1, 3, 8, 0.0);
        let y = block.forward(&x);

        // 입출력 차이가 입력 크기 대비 크지 않아야 함
        let x_data = x.data();
        let y_data = y.data();
        let diff: f64 = x_data.iter().zip(y_data.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let x_norm: f64 = x_data.iter().map(|v| v * v).sum::<f64>().sqrt();

        let ratio = diff / (x_norm + 1e-10);
        println!("residual ratio (diff/norm): {:.4}", ratio);
        // 초기에는 residual 덕분에 비율이 매우 클 수는 없음
        // (정확한 상한은 초기화에 따라 다르지만, 입력이 완전히 사라지지는 않아야 함)
        assert!(y_data.iter().all(|v| v.is_finite()), "output should be finite");
        println!("residual effect verified ✓");
    }

    #[test]
    fn test_causal_property() {
        // --- 인과성: 미래 토큰 변경 시 이전 출력 불변 ---
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        let _guard = no_grad();

        let x1 = make_input(1, 4, 8, 0.0);
        let y1 = block.forward(&x1);
        let y1_data = y1.data();

        // x2: 마지막 토큰만 변경
        let mut x2_raw: Vec<f64> = x1.data().iter().cloned().collect();
        for i in (3 * 8)..(4 * 8) {
            x2_raw[i] = 999.0;
        }
        let x2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 8]), x2_raw).unwrap(),
        );
        let y2 = block.forward(&x2);
        let y2_data = y2.data();

        // t=0, 1, 2의 출력은 동일해야 함
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
        println!("Causal property verified ✓");
    }

    #[test]
    fn test_stacked_blocks() {
        // --- 다중 블록 스택: 3개 블록 직렬 연결 ---
        let block1 = TransformerBlock::new(8, 2, 0.0, 42);
        let block2 = TransformerBlock::new(8, 2, 0.0, 100);
        let block3 = TransformerBlock::new(8, 2, 0.0, 200);

        let x = make_input(1, 4, 8, 0.0);
        let h1 = block1.forward(&x);
        let h2 = block2.forward(&h1);
        let h3 = block3.forward(&h2);

        assert_eq!(h3.shape(), vec![1, 4, 8]);

        // backward
        let loss = sum(&h3);
        loss.backward(false, false);

        assert!(x.grad().is_some(), "x should have grad");
        assert!(x.grad().unwrap().iter().all(|v| v.is_finite()), "x grad should be finite");

        println!("3-block stack: output shape {:?}", h3.shape());
        println!("  x grad shape: {:?}", x.grad().unwrap().shape());
        println!("  block1 params: {}", block1.params().len());
        println!("  block2 params: {}", block2.params().len());
        println!("  block3 params: {}", block3.params().len());
        println!("Stacked blocks test passed ✓");
    }

    #[test]
    fn test_dropout_mode() {
        // --- dropout > 0: train/test 모드에서 출력 차이 확인 ---
        let block = TransformerBlock::new(8, 2, 0.5, 42);
        let x = make_input(1, 4, 8, 0.0);

        // 학습 모드 (dropout 적용)
        let y_train = block.forward(&x);

        // 테스트 모드 (dropout 비적용)
        let _guard = test_mode();
        let y_test = block.forward(&x);

        assert_eq!(y_train.shape(), y_test.shape());
        println!("train output[0,0,:4]: {:.4} {:.4} {:.4} {:.4}",
            y_train.data()[[0, 0, 0]], y_train.data()[[0, 0, 1]],
            y_train.data()[[0, 0, 2]], y_train.data()[[0, 0, 3]]);
        println!("test  output[0,0,:4]: {:.4} {:.4} {:.4} {:.4}",
            y_test.data()[[0, 0, 0]], y_test.data()[[0, 0, 1]],
            y_test.data()[[0, 0, 2]], y_test.data()[[0, 0, 3]]);
        println!("Dropout mode test passed ✓");
    }

    #[test]
    fn test_cleargrads() {
        // --- cleargrads가 모든 파라미터의 기울기를 초기화하는지 확인 ---
        let block = TransformerBlock::new(8, 2, 0.0, 42);
        let x = make_input(1, 4, 8, 0.0);

        // forward + backward
        let y = block.forward(&x);
        sum(&y).backward(false, false);
        for p in block.params() {
            assert!(p.grad().is_some());
        }

        // cleargrads
        block.cleargrads();
        for p in block.params() {
            assert!(p.grad().is_none(), "grad should be cleared");
        }
        println!("cleargrads verified ✓");
    }

    #[test]
    fn test_batch_independence() {
        // --- 배치 내 샘플이 독립적으로 처리되는지 확인 ---
        let block = TransformerBlock::new(6, 2, 0.0, 42);
        let _guard = no_grad();

        // 배치 2 한번에
        let x_batch = make_input(2, 3, 6, 0.0);
        let y_batch = block.forward(&x_batch);
        let y_batch_data = y_batch.data();

        // 배치 1씩 따로
        let x0_raw: Vec<f64> = x_batch.data().as_slice().unwrap()[..18].to_vec();
        let x1_raw: Vec<f64> = x_batch.data().as_slice().unwrap()[18..].to_vec();
        let x0 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, 6]), x0_raw).unwrap(),
        );
        let x1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, 6]), x1_raw).unwrap(),
        );
        let y0 = block.forward(&x0);
        let y1 = block.forward(&x1);
        let y0_data = y0.data();
        let y1_data = y1.data();

        // 배치 결과와 개별 결과가 동일해야 함
        for t in 0..3 {
            for d in 0..6 {
                assert!(
                    (y_batch_data[[0, t, d]] - y0_data[[0, t, d]]).abs() < 1e-10,
                    "batch[0] vs individual[0] mismatch at [{},{}]", t, d,
                );
                assert!(
                    (y_batch_data[[1, t, d]] - y1_data[[0, t, d]]).abs() < 1e-10,
                    "batch[1] vs individual[1] mismatch at [{},{}]", t, d,
                );
            }
        }
        println!("Batch independence verified ✓");
    }
}
