// step70: Sentence Embedding — Contrastive Learning (NT-Xent)
//
// BERT 인코더 + Mean Pooling으로 문장 벡터를 추출하고,
// NT-Xent (InfoNCE) loss로 contrastive 학습:
//
//   S_ij = cos(z_a_i, z_b_j) / τ
//   L = ½ [CE(S, diag) + CE(S^T, diag)]
//
// 양의 쌍은 가깝게 (alignment), 전체 분포는 균등하게 (uniformity)
// → 의미있는 문장 벡터 공간 형성
//
// 학습 후 encode()로 projection 전 hidden을 문장 벡터로 사용
// (SimCLR: projection head는 학습 보조, 추론에서 제거)

use dezero::{AdamW, SentenceEmbedding, Variable, nt_xent_loss};

#[cfg(test)]
mod tests {
    use super::*;

    // --- 헬퍼 ---

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (norm_a * norm_b + 1e-10)
    }

    /// 토큰 ID Variable 생성: (B, T)
    fn make_token_ids(data: &[Vec<f64>]) -> Variable {
        let b = data.len();
        let t = data[0].len();
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, t]), flat).unwrap()
        )
    }

    /// 제로 세그먼트 ID: (B, T)
    fn zeros_segment(b: usize, t: usize) -> Variable {
        Variable::new(
            ndarray::ArrayD::zeros(ndarray::IxDyn(&[b, t]))
        )
    }

    /// 간단한 LCG 난수
    fn next_rng(rng: &mut u64) -> usize {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*rng >> 33) as usize)
    }

    // --- 테스트 ---

    #[test]
    fn test_forward_shape() {
        // SentenceEmbedding forward → (B, proj_dim) 출력 확인
        let vocab = 20;
        let n_embd = 16;
        let n_head = 2;
        let n_layer = 1;
        let max_seq = 8;
        let proj_dim = 8;

        let model = SentenceEmbedding::new(vocab, n_embd, n_head, n_layer, max_seq, proj_dim, 0.0, 42);

        let tokens = make_token_ids(&[
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ]);
        let segments = zeros_segment(3, 4);

        let z = model.forward(&tokens, &segments);
        println!("forward shape: {:?}", z.shape());
        assert_eq!(z.shape(), vec![3, proj_dim]);
        println!("forward shape test passed ✓");
    }

    #[test]
    fn test_encode_shape() {
        // encode()는 projection 전 (B, D) 반환
        let model = SentenceEmbedding::new(20, 16, 2, 1, 8, 8, 0.0, 42);
        let tokens = make_token_ids(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let segments = zeros_segment(2, 3);

        let h = model.encode(&tokens, &segments);
        println!("encode shape: {:?}", h.shape());
        assert_eq!(h.shape(), vec![2, 16]); // (B, n_embd)
        println!("encode shape test passed ✓");
    }

    #[test]
    fn test_backward() {
        // 모든 파라미터에 gradient가 생성되는지 확인
        let model = SentenceEmbedding::new(20, 16, 2, 1, 8, 8, 0.0, 42);

        let tok_a = make_token_ids(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let tok_b = make_token_ids(&[vec![1.0, 3.0, 2.0], vec![4.0, 6.0, 5.0]]);
        let seg = zeros_segment(2, 3);

        let z_a = model.forward(&tok_a, &seg);
        let z_b = model.forward(&tok_b, &seg);
        let loss = nt_xent_loss(&z_a, &z_b, 0.1);
        loss.backward(false, false);

        let params = model.params();
        println!("param count: {}", params.len());
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
            let g = p.grad().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "param {} grad has NaN/Inf", i);
        }
        println!("backward test passed ✓");
    }

    #[test]
    fn test_ntxent_loss_shape() {
        // NT-Xent loss: 스칼라, 양수
        let n = 4;
        let d = 8;
        let mut rng = 42u64;
        let data_a: Vec<f64> = (0..n*d).map(|_| { next_rng(&mut rng); (*&rng as f64 / u64::MAX as f64) - 0.5 }).collect();
        let data_b: Vec<f64> = (0..n*d).map(|_| { next_rng(&mut rng); (*&rng as f64 / u64::MAX as f64) - 0.5 }).collect();

        let z_a = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data_a).unwrap());
        let z_b = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data_b).unwrap());

        let loss = nt_xent_loss(&z_a, &z_b, 0.1);
        println!("NT-Xent loss shape: {:?}, value: {:.4}", loss.shape(), loss.data()[[]]);
        assert_eq!(loss.shape().len(), 0); // 스칼라
        assert!(loss.data()[[]] > 0.0, "loss should be positive");
        println!("ntxent loss shape test passed ✓");
    }

    #[test]
    fn test_ntxent_loss_gradient() {
        // NT-Xent backward: z_a, z_b 모두 gradient 생성
        let n = 3;
        let d = 4;
        let data_a: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0];
        let data_b: Vec<f64> = vec![0.9, 0.1, 0.0, 0.0,  // z_a[0]과 유사
                                     0.0, 0.8, 0.2, 0.0,  // z_a[1]과 유사
                                     0.1, 0.0, 0.9, 0.0]; // z_a[2]와 유사

        let z_a = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data_a).unwrap());
        let z_b = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data_b).unwrap());

        let loss = nt_xent_loss(&z_a, &z_b, 0.5);
        println!("loss: {:.4}", loss.data()[[]]);
        loss.backward(false, false);

        let ga = z_a.grad().expect("z_a should have grad");
        let gb = z_b.grad().expect("z_b should have grad");
        assert!(ga.iter().all(|v| v.is_finite()), "z_a grad has NaN/Inf");
        assert!(gb.iter().all(|v| v.is_finite()), "z_b grad has NaN/Inf");
        println!("z_a grad norm: {:.6}", ga.iter().map(|v| v * v).sum::<f64>().sqrt());
        println!("z_b grad norm: {:.6}", gb.iter().map(|v| v * v).sum::<f64>().sqrt());
        println!("ntxent gradient test passed ✓");
    }

    #[test]
    fn test_ntxent_perfect_alignment() {
        // 완벽한 양의 쌍 (z_a == z_b)일 때 loss가 최소에 가까운지 확인
        let n = 3;
        let d = 4;
        let data: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0, 0.0];
        let z_a = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data.clone()).unwrap());
        let z_b = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data).unwrap());

        let loss_perfect = nt_xent_loss(&z_a, &z_b, 0.5);

        // 랜덤 매핑 (z_b가 다른 순서)일 때 loss가 더 높아야 함
        let data_shuffled: Vec<f64> = vec![0.0, 1.0, 0.0, 0.0,  // z_a[1]과 매칭
                                            0.0, 0.0, 1.0, 0.0,  // z_a[2]와 매칭
                                            1.0, 0.0, 0.0, 0.0]; // z_a[0]과 매칭
        let z_a2 = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap());
        let z_b2 = Variable::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, d]), data_shuffled).unwrap());
        let loss_shuffled = nt_xent_loss(&z_a2, &z_b2, 0.5);

        println!("perfect alignment loss: {:.4}", loss_perfect.data()[[]]);
        println!("shuffled alignment loss: {:.4}", loss_shuffled.data()[[]]);
        assert!(
            loss_perfect.data()[[]] < loss_shuffled.data()[[]],
            "perfect alignment should have lower loss"
        );
        println!("perfect alignment test passed ✓");
    }

    #[test]
    fn test_loss_decreases() {
        // contrastive 학습으로 loss가 감소하는지 검증
        // 패턴: 유사 토큰을 가진 문장 쌍을 반복 학습
        let vocab = 16;
        let n_embd = 16;
        let proj_dim = 8;
        let model = SentenceEmbedding::new(vocab, n_embd, 2, 1, 6, proj_dim, 0.0, 42);
        let opt = AdamW::new(0.001, 0.0);
        let tau = 0.5;

        // 학습 데이터: 4쌍의 유사 문장
        // 쌍 내에서는 토큰이 유사, 쌍 간에는 다른 토큰
        let pairs_a = vec![
            vec![1.0, 2.0, 3.0, 4.0],  // 문장 A1
            vec![5.0, 6.0, 7.0, 8.0],  // 문장 A2
            vec![9.0, 10.0, 11.0, 12.0], // 문장 A3
            vec![13.0, 14.0, 15.0, 1.0], // 문장 A4
        ];
        let pairs_b = vec![
            vec![2.0, 1.0, 4.0, 3.0],  // B1: A1과 유사 (같은 토큰, 다른 순서)
            vec![6.0, 5.0, 8.0, 7.0],  // B2: A2와 유사
            vec![10.0, 9.0, 12.0, 11.0], // B3: A3과 유사
            vec![14.0, 13.0, 1.0, 15.0], // B4: A4와 유사
        ];

        let tok_a = make_token_ids(&pairs_a);
        let tok_b = make_token_ids(&pairs_b);
        let seg = zeros_segment(4, 4);

        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for epoch in 0..100 {
            model.cleargrads();

            let z_a = model.forward(&tok_a, &seg);
            let z_b = model.forward(&tok_b, &seg);
            let loss = nt_xent_loss(&z_a, &z_b, tau);
            loss.backward(false, false);
            opt.update(&model.params());

            let loss_val = loss.data()[[]];
            if epoch == 0 { first_loss = loss_val; }
            last_loss = loss_val;

            if epoch < 3 || (epoch + 1) % 25 == 0 {
                println!("epoch {:3} | loss {:.4}", epoch + 1, loss_val);
            }
        }

        println!("first loss: {:.4}, last loss: {:.4}", first_loss, last_loss);
        assert!(
            last_loss < first_loss * 0.8,
            "loss should decrease: {:.4} → {:.4}", first_loss, last_loss,
        );
        println!("loss decrease verified ✓");
    }

    #[test]
    fn test_similar_sentences_close() {
        // 학습 후 유사 문장의 코사인 유사도가 비유사 문장보다 높은지
        let vocab = 16;
        let n_embd = 16;
        let proj_dim = 8;
        let model = SentenceEmbedding::new(vocab, n_embd, 2, 1, 4, proj_dim, 0.0, 42);
        let opt = AdamW::new(0.001, 0.0);

        // 패턴 학습: 그룹 A(토큰 1-4), 그룹 B(토큰 9-12)
        let pairs_a = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let pairs_b = vec![
            vec![2.0, 1.0, 4.0, 3.0],  // 그룹 A와 유사
            vec![10.0, 9.0, 12.0, 11.0], // 그룹 B와 유사
        ];

        let tok_a = make_token_ids(&pairs_a);
        let tok_b = make_token_ids(&pairs_b);
        let seg = zeros_segment(2, 4);

        for _epoch in 0..150 {
            model.cleargrads();
            let z_a = model.forward(&tok_a, &seg);
            let z_b = model.forward(&tok_b, &seg);
            let loss = nt_xent_loss(&z_a, &z_b, 0.5);
            loss.backward(false, false);
            opt.update(&model.params());
        }

        // encode()로 문장 벡터 추출 (projection 전)
        let test_sents = vec![
            vec![1.0, 2.0, 3.0, 4.0],  // 그룹 A
            vec![2.0, 3.0, 4.0, 1.0],  // 그룹 A (변형)
            vec![9.0, 10.0, 11.0, 12.0], // 그룹 B
        ];
        let test_tok = make_token_ids(&test_sents);
        let test_seg = zeros_segment(3, 4);

        let _guard = dezero::no_grad();
        let vecs = model.encode(&test_tok, &test_seg);
        let vecs_data = vecs.data();
        let d = n_embd;

        let v0: Vec<f64> = (0..d).map(|j| vecs_data[[0, j]]).collect();
        let v1: Vec<f64> = (0..d).map(|j| vecs_data[[1, j]]).collect();
        let v2: Vec<f64> = (0..d).map(|j| vecs_data[[2, j]]).collect();

        let sim_same = cosine_similarity(&v0, &v1);   // A-A: 같은 그룹
        let sim_diff = cosine_similarity(&v0, &v2);   // A-B: 다른 그룹

        println!("cos(A, A') = {:.4} (same group, should be higher)", sim_same);
        println!("cos(A, B)  = {:.4} (different group)", sim_diff);

        assert!(
            sim_same > sim_diff,
            "same-group similarity ({:.4}) should exceed cross-group ({:.4})",
            sim_same, sim_diff,
        );
        println!("similar sentences close verified ✓");
    }

    #[test]
    fn test_cleargrads() {
        let model = SentenceEmbedding::new(20, 16, 2, 1, 8, 8, 0.0, 42);
        let tokens = make_token_ids(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let seg = zeros_segment(2, 3);

        let z_a = model.forward(&tokens, &seg);
        let z_b = model.forward(&tokens, &seg);
        let loss = nt_xent_loss(&z_a, &z_b, 0.1);
        loss.backward(false, false);

        for p in model.params() {
            assert!(p.grad().is_some());
        }

        model.cleargrads();
        for p in model.params() {
            assert!(p.grad().is_none(), "grad should be cleared");
        }
        println!("cleargrads verified ✓");
    }
}
