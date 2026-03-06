// step69: Word2Vec — Skip-gram + Negative Sampling
//
// 단어를 밀집 벡터(dense vector)로 표현하는 핵심 기법 (Mikolov et al., 2013)
//
// Skip-gram: 중심 단어로 주변 문맥 단어를 예측
//   "The cat sat on the mat" → (sat, cat), (sat, on), ...
//
// Negative Sampling: 전체 어휘 softmax 대신 K개 부정 샘플만 사용
//   Loss = -log σ(u_pos · v) - Σ_k log σ(-u_neg_k · v)
//
// 학습 후 W_in이 단어 벡터로 사용됨
// 의미가 유사한 단어끼리 벡터 공간에서 가까워짐

use dezero::{AdamW, Word2Vec, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    // --- 헬퍼 함수 ---

    /// 코퍼스에서 (center, context) Skip-gram 쌍 생성
    fn build_skip_gram_pairs(corpus: &[usize], window: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..corpus.len() {
            for j in 1..=window {
                if i >= j {
                    pairs.push((corpus[i], corpus[i - j]));
                }
                if i + j < corpus.len() {
                    pairs.push((corpus[i], corpus[i + j]));
                }
            }
        }
        pairs
    }

    /// Unigram 분포 (0.75승) 기반 부정 샘플링 테이블
    /// 빈도가 높은 단어는 많이, 낮은 단어도 적당히 샘플링
    fn build_negative_table(word_counts: &[usize], table_size: usize) -> Vec<usize> {
        let total: f64 = word_counts.iter().map(|&c| (c as f64).powf(0.75)).sum();
        let mut table = Vec::with_capacity(table_size);
        for (idx, &count) in word_counts.iter().enumerate() {
            let prob = (count as f64).powf(0.75) / total;
            let n = (prob * table_size as f64).round() as usize;
            for _ in 0..n.min(table_size - table.len()) {
                table.push(idx);
            }
        }
        // 부족분 채우기
        while table.len() < table_size {
            table.push(word_counts.len() - 1);
        }
        table.truncate(table_size);
        table
    }

    /// 부정 샘플 K개 추출 (positive와 다른 단어만)
    fn sample_negatives(table: &[usize], k: usize, positive: usize, rng: &mut u64) -> Vec<usize> {
        let mut negs = Vec::with_capacity(k);
        while negs.len() < k {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = ((*rng >> 33) as usize) % table.len();
            let word = table[idx];
            if word != positive {
                negs.push(word);
            }
        }
        negs
    }

    /// 코사인 유사도
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (norm_a * norm_b + 1e-10)
    }

    /// 학습용 배치 Variable 생성
    fn make_batch(
        pairs: &[(usize, usize)],
        table: &[usize],
        k: usize,
        rng: &mut u64,
    ) -> (Variable, Variable, Variable) {
        let n = pairs.len();
        let centers: Vec<f64> = pairs.iter().map(|&(c, _)| c as f64).collect();
        let contexts: Vec<f64> = pairs.iter().map(|&(_, ctx)| ctx as f64).collect();
        let mut neg_data = Vec::with_capacity(n * k);
        for &(_, ctx) in pairs {
            let negs = sample_negatives(table, k, ctx, rng);
            for neg in negs {
                neg_data.push(neg as f64);
            }
        }

        let center_var = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), centers).unwrap(),
        );
        let context_var = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), contexts).unwrap(),
        );
        let neg_var = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, k]), neg_data).unwrap(),
        );
        (center_var, context_var, neg_var)
    }

    #[test]
    fn test_forward_shape() {
        // --- forward: 스칼라 loss 반환 확인 ---
        let model = Word2Vec::new(10, 4, 42);
        let center = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 1.0, 2.0]).unwrap(),
        );
        let context = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![3.0, 4.0, 5.0]).unwrap(),
        );
        let negatives = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3, 5]),
                (0..15).map(|i| (i % 10) as f64).collect(),
            ).unwrap(),
        );
        let loss = model.forward(&center, &context, &negatives);

        println!("loss shape: {:?}", loss.shape());
        println!("loss value: {:.4}", loss.data()[[]]);
        assert_eq!(loss.shape().len(), 0); // 스칼라
        assert!(loss.data()[[]] > 0.0, "loss should be positive");
        println!("forward shape test passed ✓");
    }

    #[test]
    fn test_backward() {
        // --- backward: W_in, W_out 모두 gradient 생성 ---
        let model = Word2Vec::new(10, 4, 42);
        let center = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        );
        let context = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
        );
        let negatives = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![5.0, 6.0, 7.0, 8.0, 9.0, 2.0],
            ).unwrap(),
        );
        let loss = model.forward(&center, &context, &negatives);
        loss.backward(false, false);

        let params = model.params();
        println!("param count: {}", params.len());
        assert_eq!(params.len(), 2); // W_in, W_out
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
            let g = p.grad().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "param {} grad has NaN/Inf", i);
            println!("  param {} shape: {:?}, grad norm: {:.6}",
                i, p.shape(),
                g.iter().map(|v| v * v).sum::<f64>().sqrt());
        }
        println!("backward test passed ✓");
    }

    #[test]
    fn test_loss_decreases() {
        // --- 학습: "you say goodbye and i say hello" 코퍼스 ---
        // you=0, say=1, goodbye=2, and=3, i=4, hello=5
        let corpus = vec![0, 1, 2, 3, 4, 1, 5];
        let word_counts = vec![1, 2, 1, 1, 1, 1]; // say=2
        let vocab_size = 6;

        let pairs = build_skip_gram_pairs(&corpus, 1);
        let table = build_negative_table(&word_counts, 100);
        println!("skip-gram pairs: {}", pairs.len());
        println!("pairs: {:?}", pairs);

        let model = Word2Vec::new(vocab_size, 8, 42);
        let opt = AdamW::new(0.01, 0.0);
        let k = 5;
        let mut rng = 123u64;

        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for epoch in 0..200 {
            let mut total_loss = 0.0;
            let mut count = 0;

            for &(center, context) in &pairs {
                model.cleargrads();

                let negs = sample_negatives(&table, k, context, &mut rng);
                let c_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![center as f64]).unwrap(),
                );
                let ctx_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![context as f64]).unwrap(),
                );
                let neg_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, k]),
                        negs.iter().map(|&n| n as f64).collect(),
                    ).unwrap(),
                );

                let loss = model.forward(&c_var, &ctx_var, &neg_var);
                loss.backward(false, false);
                opt.update(&model.params());

                total_loss += loss.data()[[]];
                count += 1;
            }

            let avg_loss = total_loss / count as f64;
            if epoch == 0 { first_loss = avg_loss; }
            last_loss = avg_loss;

            if epoch < 3 || (epoch + 1) % 50 == 0 {
                println!("epoch {:3} | loss {:.4}", epoch + 1, avg_loss);
            }
        }

        println!("first loss: {:.4}, last loss: {:.4}", first_loss, last_loss);
        assert!(
            last_loss < first_loss * 0.5,
            "loss should decrease significantly: {} → {}",
            first_loss, last_loss,
        );
        println!("loss decrease verified ✓");
    }

    #[test]
    fn test_similar_words_close() {
        // --- 유사 문맥 단어는 유사한 벡터를 갖는지 검증 ---
        // 패턴: "a X b Y a X b Y ..." (0 2 1 3 0 2 1 3 ...)
        // a(0)와 b(1)은 유사한 문맥을 공유: 양쪽에 X(2)/Y(3)
        // X(2)와 Y(3)도 유사: 양쪽에 a(0)/b(1)
        let pattern: Vec<usize> = (0..40).map(|i| match i % 4 {
            0 => 0, // a
            1 => 2, // X
            2 => 1, // b
            _ => 3, // Y
        }).collect();

        let word_counts = vec![10, 10, 10, 10]; // 균등
        let vocab_size = 4;
        let embed_dim = 8;

        let pairs = build_skip_gram_pairs(&pattern, 1);
        let table = build_negative_table(&word_counts, 100);

        let model = Word2Vec::new(vocab_size, embed_dim, 42);
        let opt = AdamW::new(0.01, 0.0);
        let k = 2; // 어휘가 작으므로 부정 샘플도 적게
        let mut rng = 999u64;

        // 학습
        for _epoch in 0..300 {
            for &(center, context) in &pairs {
                model.cleargrads();
                let negs = sample_negatives(&table, k, context, &mut rng);
                let c_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![center as f64]).unwrap(),
                );
                let ctx_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![context as f64]).unwrap(),
                );
                let neg_var = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, k]),
                        negs.iter().map(|&n| n as f64).collect(),
                    ).unwrap(),
                );

                let loss = model.forward(&c_var, &ctx_var, &neg_var);
                loss.backward(false, false);
                opt.update(&model.params());
            }
        }

        // 벡터 추출
        let vectors = model.get_word_vectors();
        let d = embed_dim;
        let vec_a: Vec<f64> = (0..d).map(|j| vectors[[0, j]]).collect();
        let vec_b: Vec<f64> = (0..d).map(|j| vectors[[1, j]]).collect();
        let vec_x: Vec<f64> = (0..d).map(|j| vectors[[2, j]]).collect();
        let vec_y: Vec<f64> = (0..d).map(|j| vectors[[3, j]]).collect();

        let sim_ab = cosine_similarity(&vec_a, &vec_b);
        let sim_xy = cosine_similarity(&vec_x, &vec_y);
        let sim_ax = cosine_similarity(&vec_a, &vec_x);

        println!("cos(a, b) = {:.4} (same role, should be high)", sim_ab);
        println!("cos(X, Y) = {:.4} (same role, should be high)", sim_xy);
        println!("cos(a, X) = {:.4} (different role)", sim_ax);

        // a와 b는 유사해야 함 (같은 문맥 패턴)
        assert!(
            sim_ab > sim_ax,
            "a-b similarity ({:.4}) should be higher than a-X ({:.4})",
            sim_ab, sim_ax,
        );
        println!("similar words close verified ✓");
    }

    #[test]
    fn test_cleargrads() {
        let model = Word2Vec::new(10, 4, 42);
        let center = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        );
        let context = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
        );
        let negatives = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![5.0, 6.0, 7.0, 8.0, 9.0, 2.0],
            ).unwrap(),
        );

        let loss = model.forward(&center, &context, &negatives);
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

    #[test]
    fn test_negative_table_distribution() {
        // --- unigram^0.75 테이블이 기대한 분포를 가지는지 ---
        let counts = vec![100, 10, 1]; // 빈도 비율 100:10:1
        let table = build_negative_table(&counts, 10000);

        let mut freq = vec![0usize; 3];
        for &w in &table {
            freq[w] += 1;
        }

        // 0.75승 기대 비율: 100^0.75 : 10^0.75 : 1^0.75
        //   = 31.62 : 5.62 : 1.0 → 82.7% : 14.7% : 2.6%
        let total = freq.iter().sum::<usize>() as f64;
        let pct: Vec<f64> = freq.iter().map(|&f| f as f64 / total * 100.0).collect();

        println!("freq distribution: {:.1}% : {:.1}% : {:.1}%", pct[0], pct[1], pct[2]);

        // 빈도 높은 단어가 가장 많이, 낮은 단어도 0은 아님
        assert!(pct[0] > pct[1], "high freq word should appear more");
        assert!(pct[1] > pct[2], "medium freq word should appear more than rare");
        assert!(pct[2] > 0.5, "rare word should still have some representation");
        println!("negative table distribution verified ✓");
    }
}
