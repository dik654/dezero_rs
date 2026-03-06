// step71: Brute Force 벡터 검색
//
// Phase 4 (벡터 검색)의 첫 번째 스텝.
// 4가지 유사도/거리 메트릭과 BruteForceIndex로 정확한 최근접 탐색 구현.
//
// 벡터 검색은 추론 전용 — Variable/autograd 없이 순수 ndarray 연산.
// Brute force는 O(N·D)/query로 정확한 exact NN의 기준.
// 이후 IVF, PQ, HNSW의 recall을 검증하는 ground truth 역할.

use dezero::{
    cosine_similarity_vec, dot_product_vec, l2_distance, l1_distance,
    BruteForceIndex, Metric,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ============================
    // 유사도/거리 함수 테스트
    // ============================

    #[test]
    fn test_cosine_similarity() {
        // 동일 방향 → 1.0
        assert!((cosine_similarity_vec(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        // 직교 → 0.0
        assert!(cosine_similarity_vec(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        // 반대 방향 → -1.0
        assert!((cosine_similarity_vec(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // 스케일 불변: [1,2,3]과 [2,4,6]은 같은 방향
        assert!((cosine_similarity_vec(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-6);
        // 알려진 값: cos([1,1], [1,0]) = 1/√2 ≈ 0.7071
        let val = cosine_similarity_vec(&[1.0, 1.0], &[1.0, 0.0]);
        assert!((val - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);

        println!("cosine_similarity: 동일={:.4}, 직교={:.4}, 반대={:.4}, 스케일불변={:.4}, 1/√2={:.4}",
            cosine_similarity_vec(&[1.0, 0.0], &[1.0, 0.0]),
            cosine_similarity_vec(&[1.0, 0.0], &[0.0, 1.0]),
            cosine_similarity_vec(&[1.0, 0.0], &[-1.0, 0.0]),
            cosine_similarity_vec(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]),
            val,
        );
    }

    #[test]
    fn test_dot_product() {
        // [1,2,3]·[4,5,6] = 4+10+18 = 32
        assert!((dot_product_vec(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
        // 직교
        assert!(dot_product_vec(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-10);
        // 영벡터
        assert!(dot_product_vec(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]).abs() < 1e-10);

        println!("dot_product: [1,2,3]·[4,5,6]={:.1}, 직교={:.1}",
            dot_product_vec(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            dot_product_vec(&[1.0, 0.0], &[0.0, 1.0]),
        );
    }

    #[test]
    fn test_l2_distance() {
        // 동일 벡터 → 0
        assert!(l2_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-10);
        // 알려진 값: √((4-1)²+(0-0)²) = 3
        assert!((l2_distance(&[1.0, 0.0], &[4.0, 0.0]) - 3.0).abs() < 1e-10);
        // 삼각 부등식: d(a,c) ≤ d(a,b) + d(b,c)
        let a = &[0.0, 0.0];
        let b = &[1.0, 0.0];
        let c = &[1.0, 1.0];
        assert!(l2_distance(a, c) <= l2_distance(a, b) + l2_distance(b, c) + 1e-10);

        println!("l2_distance: 동일={:.4}, [1,0]→[4,0]={:.1}, 삼각부등식 OK",
            l2_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            l2_distance(&[1.0, 0.0], &[4.0, 0.0]),
        );
    }

    #[test]
    fn test_l1_distance() {
        // 동일 벡터 → 0
        assert!(l1_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-10);
        // |1-4| + |2-5| + |3-6| = 9
        assert!((l1_distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 9.0).abs() < 1e-10);
        // L1 ≥ L2 (Jensen's inequality)
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 0.0, 1.0];
        assert!(l1_distance(a, b) >= l2_distance(a, b) - 1e-10);

        println!("l1_distance: 동일={:.4}, [1,2,3]→[4,5,6]={:.1}, L1({:.4}) ≥ L2({:.4})",
            l1_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            l1_distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            l1_distance(a, b), l2_distance(a, b),
        );
    }

    // ============================
    // BruteForceIndex 테스트
    // ============================

    #[test]
    fn test_index_add_and_len() {
        let mut index = BruteForceIndex::new(3);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        index.add(&[1.0, 0.0, 0.0], "x-axis");
        index.add(&[0.0, 1.0, 0.0], "y-axis");
        index.add(&[0.0, 0.0, 1.0], "z-axis");

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
        assert_eq!(index.get(0), &[1.0, 0.0, 0.0]);
        assert_eq!(index.dim(), 3);

        println!("index: len={}, dim={}, vec[0]={:?}", index.len(), index.dim(), index.get(0));
    }

    #[test]
    fn test_search_cosine() {
        let mut index = BruteForceIndex::new(3);
        index.add(&[1.0, 0.0, 0.0], "doc_a");   // x축
        index.add(&[0.0, 1.0, 0.0], "doc_b");   // y축
        index.add(&[0.9, 0.1, 0.0], "doc_c");   // x축 근처
        index.add(&[0.0, 0.0, 1.0], "doc_d");   // z축

        // x축 쿼리: doc_a(완벽) > doc_c(거의 같은 방향)
        let results = index.search(&[1.0, 0.0, 0.0], 2, Metric::Cosine);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].2, "doc_a");
        assert_eq!(results[1].2, "doc_c");
        assert!((results[0].1 - 1.0).abs() < 1e-6);

        // k > N이면 전체 반환
        let all = index.search(&[1.0, 0.0, 0.0], 100, Metric::Cosine);
        assert_eq!(all.len(), 4);

        println!("search_cosine top-2:");
        for (rank, (idx, score, label)) in results.iter().enumerate() {
            println!("  #{}: {} (idx={}, cos={:.4})", rank + 1, label, idx, score);
        }
    }

    #[test]
    fn test_search_all_metrics() {
        let mut index = BruteForceIndex::new(2);
        index.add(&[1.0, 0.0], "close");
        index.add(&[0.0, 1.0], "far");
        index.add(&[0.7, 0.7], "medium");

        let query = &[1.0, 0.1];

        // 모든 메트릭에서 "close"가 1위
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::L2, Metric::L1] {
            let r = index.search(query, 3, metric);
            assert_eq!(r[0].2, "close", "metric {:?}: expected 'close' as #1, got '{}'", metric, r[0].2);
            println!("{:?}: #1={} ({:.4}), #2={} ({:.4}), #3={} ({:.4})",
                metric, r[0].2, r[0].1, r[1].2, r[1].1, r[2].2, r[2].1);
        }
    }

    #[test]
    fn test_batch_search() {
        let mut index = BruteForceIndex::new(2);
        index.add(&[1.0, 0.0], "east");
        index.add(&[0.0, 1.0], "north");
        index.add(&[-1.0, 0.0], "west");

        // 2개 쿼리 동시 검색
        let queries = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 2]),
            vec![0.9, 0.1,    // → east
                 -0.8, 0.1],  // → west
        ).unwrap();

        let results = index.batch_search(&queries, 1, Metric::Cosine);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].2, "east");
        assert_eq!(results[1][0].2, "west");

        println!("batch_search: query0→{}, query1→{}", results[0][0].2, results[1][0].2);
    }

    #[test]
    fn test_empty_index() {
        let index = BruteForceIndex::new(4);
        assert!(index.is_empty());
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, Metric::Cosine);
        assert!(results.is_empty());

        println!("empty index search: {} results", results.len());
    }

    #[test]
    fn test_add_batch() {
        let mut index = BruteForceIndex::new(3);

        let vecs = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 3]),
            vec![1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0],
        ).unwrap();
        let labels = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        index.add_batch(&vecs, &labels);

        assert_eq!(index.len(), 3);
        let r = index.search(&[1.0, 0.0, 0.0], 1, Metric::Cosine);
        assert_eq!(r[0].2, "x");

        println!("add_batch: len={}, search([1,0,0])→{}", index.len(), r[0].2);
    }

    #[test]
    fn test_pipeline_with_embedding() {
        use dezero::{SentenceEmbedding, AdamW, Variable, nt_xent_loss, no_grad};

        // --- 헬퍼 ---
        fn make_token_ids(data: &[Vec<f64>]) -> Variable {
            let batch = data.len();
            let seq_len = data[0].len();
            let flat: Vec<f64> = data.iter().flat_map(|v| v.iter().copied()).collect();
            Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[batch, seq_len]), flat).unwrap()
            )
        }
        fn zeros_segment(batch: usize, seq_len: usize) -> Variable {
            Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[batch, seq_len])))
        }

        // 1. 모델 학습 (contrastive)
        let vocab = 16;
        let n_embd = 16;
        let model = SentenceEmbedding::new(vocab, n_embd, 2, 1, 6, 8, 0.0, 42);
        let opt = AdamW::new(0.001, 0.0);

        // 그룹 A: 토큰 [1,2,3,4] 계열, 그룹 B: 토큰 [9,10,11,12] 계열
        let pairs_a = vec![vec![1.0, 2.0, 3.0, 4.0], vec![9.0, 10.0, 11.0, 12.0]];
        let pairs_b = vec![vec![2.0, 1.0, 4.0, 3.0], vec![10.0, 9.0, 12.0, 11.0]];
        let tok_a = make_token_ids(&pairs_a);
        let tok_b = make_token_ids(&pairs_b);
        let seg = zeros_segment(2, 4);

        for _ in 0..100 {
            model.cleargrads();
            let z_a = model.forward(&tok_a, &seg);
            let z_b = model.forward(&tok_b, &seg);
            let loss = nt_xent_loss(&z_a, &z_b, 0.5);
            loss.backward(false, false);
            opt.update(&model.params());
        }

        // 2. 인덱스 구축
        let _guard = no_grad();
        let corpus_tokens = make_token_ids(&[
            vec![1.0, 2.0, 3.0, 4.0],   // 그룹 A
            vec![9.0, 10.0, 11.0, 12.0], // 그룹 B
            vec![2.0, 3.0, 4.0, 1.0],   // 그룹 A 변형
        ]);
        let corpus_seg = zeros_segment(3, 4);
        let embeddings = model.encode(&corpus_tokens, &corpus_seg);
        let emb_data = embeddings.data();

        let mut index = BruteForceIndex::new(n_embd);
        let labels = vec!["A_original".to_string(), "B_original".to_string(), "A_variant".to_string()];
        index.add_batch(&emb_data, &labels);
        assert_eq!(index.len(), 3);

        // 3. 쿼리: 그룹 A 변형 문장으로 검색
        let query_tok = make_token_ids(&[vec![3.0, 4.0, 1.0, 2.0]]);
        let query_seg = zeros_segment(1, 4);
        let query_vec = model.encode(&query_tok, &query_seg);
        let qv = query_vec.data();
        let q: Vec<f64> = (0..n_embd).map(|j| qv[[0, j]]).collect();

        let results = index.search(&q, 3, Metric::Cosine);
        println!("\n=== Pipeline: encode → index → search ===");
        for (rank, (idx, score, label)) in results.iter().enumerate() {
            println!("  #{}: {} (idx={}, cosine={:.4})", rank + 1, label, idx, score);
        }

        // 그룹 A 문장이 top-2에 포함
        let top2_labels: Vec<&str> = results[..2].iter().map(|(_, _, l)| l.as_str()).collect();
        assert!(
            top2_labels.contains(&"A_original") || top2_labels.contains(&"A_variant"),
            "top-2 should contain group A sentences, got {:?}", top2_labels,
        );
    }
}
