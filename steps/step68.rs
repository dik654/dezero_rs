// step68: BERT — Bidirectional Encoder Transformer (MLM)
//
// GPT(step 67)와 95% 동일한 구조에서 핵심 차이:
//   1. 양방향 어텐션 (causal mask 제거 → 미래 토큰도 참조)
//   2. Segment embedding (문장 A=0 / B=1 구분)
//   3. MLM (Masked Language Modeling) — [MASK] 위치의 원래 토큰을 예측
//
// 핵심 인사이트: causal mask 하나를 빼면 왜 양방향이 되는가
// → softmax(QK^T)에서 하삼각 마스크가 없으면, 모든 위치가
//   모든 다른 위치를 attend → 양방향 문맥 활용

use dezero::{
    no_grad, reshape, masked_softmax_cross_entropy, softmax_cross_entropy_simple,
    sum, test_mode,
    AdamW, BERT, Variable,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ids(b: usize, t: usize, vocab_size: usize, seed: u64) -> Variable {
        let n = b * t;
        let mut rng = seed;
        let data: Vec<f64> = (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) % vocab_size as u64) as f64
        }).collect();
        Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, t]), data).unwrap(),
        )
    }

    fn make_segments(b: usize, t: usize, split: usize) -> Variable {
        // 앞 split개는 segment 0, 나머지는 segment 1
        let data: Vec<f64> = (0..b).flat_map(|_| {
            (0..t).map(|i| if i < split { 0.0 } else { 1.0 })
        }).collect();
        Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, t]), data).unwrap(),
        )
    }

    fn zeros_segments(b: usize, t: usize) -> Variable {
        Variable::new(
            ndarray::ArrayD::zeros(ndarray::IxDyn(&[b, t]))
        )
    }

    #[test]
    fn test_forward_shape() {
        // --- forward shape 검증: (B,T) → (B,T,V) ---
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let ids = make_ids(2, 4, 10, 0);
        let segs = zeros_segments(2, 4);
        let logits = bert.forward(&ids, &segs);

        println!("input shape: {:?}", ids.shape());
        println!("logits shape: {:?}", logits.shape());
        assert_eq!(logits.shape(), vec![2, 4, 10]);
    }

    #[test]
    fn test_backward() {
        // --- backward: 모든 파라미터에 기울기 생성 ---
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let ids = make_ids(1, 4, 10, 0);
        let segs = zeros_segments(1, 4);
        let logits = bert.forward(&ids, &segs);

        let logits_2d = reshape(&logits, &[4, 10]);
        let targets: Vec<usize> = vec![3, 5, 7, 1];
        let loss = softmax_cross_entropy_simple(&logits_2d, &targets);
        loss.backward(false, false);

        let params = bert.params();
        println!("param count: {}", params.len());
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
        }
        for (i, p) in params.iter().enumerate() {
            let g = p.grad().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "param {} grad has NaN/Inf", i);
        }
        println!("backward: all {} params have finite grads ✓", params.len());
    }

    #[test]
    fn test_bidirectional_property() {
        // --- 양방향성: 미래 토큰 변경 시 이전 출력도 변경됨 (GPT와 반대) ---
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let _guard = no_grad();

        let ids1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 4]),
                vec![1.0, 3.0, 5.0, 7.0],
            ).unwrap(),
        );
        let segs = zeros_segments(1, 4);
        let logits1 = bert.forward(&ids1, &segs);
        let l1 = logits1.data();

        // 마지막 토큰만 변경
        let ids2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 4]),
                vec![1.0, 3.0, 5.0, 2.0], // 7→2
            ).unwrap(),
        );
        let logits2 = bert.forward(&ids2, &segs);
        let l2 = logits2.data();

        // 양방향이므로 t=0,1,2의 logits도 달라야 함 (GPT와 반대!)
        let mut any_diff_early = false;
        for t in 0..3 {
            for v in 0..10 {
                if (l1[[0, t, v]] - l2[[0, t, v]]).abs() > 1e-10 {
                    any_diff_early = true;
                    break;
                }
            }
            if any_diff_early { break; }
        }
        assert!(
            any_diff_early,
            "BERT should be bidirectional: changing last token should affect earlier positions"
        );

        // t=3도 당연히 달라야 함
        let mut any_diff_last = false;
        for v in 0..10 {
            if (l1[[0, 3, v]] - l2[[0, 3, v]]).abs() > 1e-10 {
                any_diff_last = true;
                break;
            }
        }
        assert!(any_diff_last, "position 3 logits should differ");
        println!("Bidirectional property verified ✓");
    }

    #[test]
    fn test_segment_embedding() {
        // --- 다른 segment ID → 다른 출력 ---
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let _guard = no_grad();

        let ids = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 4]),
                vec![1.0, 2.0, 3.0, 4.0],
            ).unwrap(),
        );

        // 모두 segment 0
        let seg_all0 = zeros_segments(1, 4);
        let logits0 = bert.forward(&ids, &seg_all0);
        let l0 = logits0.data();

        // 앞 2개 segment 0, 뒤 2개 segment 1
        let seg_mixed = make_segments(1, 4, 2);
        let logits1 = bert.forward(&ids, &seg_mixed);
        let l1 = logits1.data();

        // 같은 토큰이지만 segment가 다르면 출력이 달라야 함
        let mut any_diff = false;
        for t in 0..4 {
            for v in 0..10 {
                if (l0[[0, t, v]] - l1[[0, t, v]]).abs() > 1e-10 {
                    any_diff = true;
                    break;
                }
            }
            if any_diff { break; }
        }
        assert!(any_diff, "different segment IDs should produce different outputs");
        println!("Segment embedding verified ✓");
    }

    #[test]
    fn test_masked_cross_entropy() {
        // --- masked_softmax_cross_entropy: 마스크된 위치만 loss 계산 ---
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let ids = make_ids(1, 4, 10, 0);
        let segs = zeros_segments(1, 4);
        let logits = bert.forward(&ids, &segs);
        let logits_2d = reshape(&logits, &[4, 10]);

        let targets: Vec<usize> = vec![3, 5, 7, 1];
        // 위치 1, 3만 마스크 (50% 마스킹)
        let mask = vec![false, true, false, true];
        let loss = masked_softmax_cross_entropy(&logits_2d, &targets, &mask);

        let loss_val: f64 = loss.data().iter().next().copied().unwrap();
        println!("masked loss: {:.4}", loss_val);
        assert!(loss_val > 0.0, "loss should be positive");
        assert!(loss_val.is_finite(), "loss should be finite");

        // 전체 loss와 비교 (마스크된 것은 다를 수 있음)
        let full_loss = softmax_cross_entropy_simple(&logits_2d, &targets);
        let full_val: f64 = full_loss.data().iter().next().copied().unwrap();
        println!("full loss: {:.4}, masked loss: {:.4}", full_val, loss_val);
        println!("Masked cross entropy verified ✓");
    }

    #[test]
    fn test_mlm_training() {
        // --- MLM 학습: 마스크된 위치의 원래 토큰을 예측, loss 감소 확인 ---
        // 패턴: "abcabc..." 반복 텍스트에서 일부를 마스킹하고 복원
        let vocab_size = 5; // a=0, b=1, c=2, d=3, [MASK]=4
        let mask_token = 4;

        // 반복 패턴: 0,1,2,0,1,2,...
        let pattern: Vec<usize> = (0..24).map(|i| i % 3).collect();

        let bert = BERT::new(vocab_size, 16, 2, 2, 32, 0.0, 42);
        let opt = AdamW::new(0.003, 0.0);

        let seq_len = 6;
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for epoch in 0..100 {
            let mut total_loss = 0.0;
            let mut count = 0;

            let mut offset = 0;
            while offset + seq_len <= pattern.len() {
                bert.cleargrads();

                let original: Vec<usize> = pattern[offset..offset + seq_len].to_vec();
                // 짝수 인덱스를 마스킹
                let mask: Vec<bool> = (0..seq_len).map(|i| i % 2 == 0).collect();
                let input: Vec<f64> = (0..seq_len).map(|i| {
                    if mask[i] { mask_token as f64 } else { original[i] as f64 }
                }).collect();

                let ids = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, seq_len]), input,
                    ).unwrap(),
                );
                let segs = zeros_segments(1, seq_len);
                let logits = bert.forward(&ids, &segs); // (1, seq_len, V)
                let logits_2d = reshape(&logits, &[seq_len, vocab_size]);
                let loss = masked_softmax_cross_entropy(&logits_2d, &original, &mask);
                loss.backward(false, false);
                opt.update(&bert.params());

                let loss_val: f64 = loss.data().iter().next().copied().unwrap();
                total_loss += loss_val;
                count += 1;
                offset += seq_len;
            }

            let avg_loss = total_loss / count as f64;
            if epoch == 0 { first_loss = avg_loss; }
            last_loss = avg_loss;

            if epoch < 3 || (epoch + 1) % 25 == 0 {
                println!("epoch {:3} | loss {:.4}", epoch + 1, avg_loss);
            }
        }

        println!("first loss: {:.4}, last loss: {:.4}", first_loss, last_loss);
        assert!(
            last_loss < first_loss * 0.5,
            "loss should decrease significantly: {} → {}",
            first_loss, last_loss,
        );
        println!("MLM training: loss decreased ✓");
    }

    #[test]
    fn test_mlm_prediction_accuracy() {
        // --- 학습 후 마스크된 토큰을 올바르게 예측하는지 확인 ---
        let vocab_size = 4; // a=0, b=1, c=2, [MASK]=3
        let mask_token = 3;

        // 단순 패턴: 항상 0,1,2 반복
        let pattern: Vec<usize> = (0..18).map(|i| i % 3).collect();

        let bert = BERT::new(vocab_size, 16, 2, 2, 32, 0.0, 42);
        let opt = AdamW::new(0.005, 0.0);

        let seq_len = 6;

        // 학습
        for _epoch in 0..200 {
            let mut offset = 0;
            while offset + seq_len <= pattern.len() {
                bert.cleargrads();

                let original: Vec<usize> = pattern[offset..offset + seq_len].to_vec();
                let mask: Vec<bool> = (0..seq_len).map(|i| i % 3 == 0).collect(); // 매 3번째 마스킹
                let input: Vec<f64> = (0..seq_len).map(|i| {
                    if mask[i] { mask_token as f64 } else { original[i] as f64 }
                }).collect();

                let ids = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, seq_len]), input,
                    ).unwrap(),
                );
                let segs = zeros_segments(1, seq_len);
                let logits = bert.forward(&ids, &segs);
                let logits_2d = reshape(&logits, &[seq_len, vocab_size]);
                let loss = masked_softmax_cross_entropy(&logits_2d, &original, &mask);
                loss.backward(false, false);
                opt.update(&bert.params());

                offset += seq_len;
            }
        }

        // 예측 테스트: [MASK],1,2,[MASK],1,2 → 마스크 위치는 0이어야 함
        let _guard = no_grad();
        let _test = test_mode();

        let input: Vec<f64> = vec![mask_token as f64, 1.0, 2.0, mask_token as f64, 1.0, 2.0];
        let ids = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, seq_len]), input,
            ).unwrap(),
        );
        let segs = zeros_segments(1, seq_len);
        let logits = bert.forward(&ids, &segs);
        let logits_data = logits.data();

        // 마스크 위치(0, 3)에서 argmax가 0이어야 함
        for pos in [0, 3] {
            let mut best = 0;
            let mut best_val = f64::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = logits_data[[0, pos, v]];
                if val > best_val {
                    best_val = val;
                    best = v;
                }
            }
            println!("position {} prediction: {} (expected 0)", pos, best);
            assert_eq!(best, 0, "masked position {} should predict 0", pos);
        }
        println!("MLM prediction accuracy verified ✓");
    }

    #[test]
    fn test_cleargrads() {
        let bert = BERT::new(10, 8, 2, 2, 16, 0.0, 42);
        let ids = make_ids(1, 4, 10, 0);
        let segs = zeros_segments(1, 4);

        let logits = bert.forward(&ids, &segs);
        sum(&logits).backward(false, false);
        for p in bert.params() {
            assert!(p.grad().is_some());
        }

        bert.cleargrads();
        for p in bert.params() {
            assert!(p.grad().is_none(), "grad should be cleared");
        }
        println!("cleargrads verified ✓");
    }

    #[test]
    fn test_batch_independence() {
        // --- 배치 내 샘플이 독립적으로 처리되는지 확인 ---
        let bert = BERT::new(10, 6, 2, 1, 16, 0.0, 42);
        let _guard = no_grad();

        // 배치 2 한번에
        let ids_batch = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ).unwrap(),
        );
        let segs_batch = zeros_segments(2, 3);
        let y_batch = bert.forward(&ids_batch, &segs_batch);
        let y_batch_data = y_batch.data();

        // 배치 1씩 따로
        let ids0 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0],
            ).unwrap(),
        );
        let ids1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 3]), vec![4.0, 5.0, 6.0],
            ).unwrap(),
        );
        let segs0 = zeros_segments(1, 3);
        let y0 = bert.forward(&ids0, &segs0);
        let y1 = bert.forward(&ids1, &segs0);
        let y0_data = y0.data();
        let y1_data = y1.data();

        for t in 0..3 {
            for d in 0..10 {
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
