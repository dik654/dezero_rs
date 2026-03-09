// step73: Product Quantization (PQ) 근사 벡터 검색
//
// Phase 4 (벡터 검색)의 세 번째 스텝.
// D차원 벡터를 M개 서브공간으로 분할, 각 서브공간에서 K-means 양자화.
// N·D·8 바이트 → N·M 바이트로 압축. ADC로 검색: O(N·D) → O(M·K'·ds + N·M).
//
// Brute force 대비 메모리를 수십 배 절약하면서 높은 recall 유지.
// 정확도/압축 트레이드오프: M이 작을수록 압축률↑ recall↓, M이 클수록 반대.

use dezero::{
    l2_distance,
    BruteForceIndex, PQIndex, Metric,
};
use std::collections::HashSet;

#[cfg(test)]
mod tests {
    use super::*;

    // --- 헬퍼 ---

    /// LCG로 N개의 D차원 랜덤 벡터 생성
    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = seed;
        (0..n).map(|_| {
            (0..dim).map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng >> 11) as f64 / (1u64 << 53) as f64
            }).collect()
        }).collect()
    }

    /// Recall@k: 근사 결과 중 정확한 결과와 일치하는 비율
    fn recall_at_k(approx: &[(usize, f64, String)], exact: &[(usize, f64, String)]) -> f64 {
        if exact.is_empty() { return 1.0; }
        let exact_ids: HashSet<usize> = exact.iter().map(|(i, _, _)| *i).collect();
        let hits = approx.iter().filter(|(i, _, _)| exact_ids.contains(i)).count();
        hits as f64 / exact.len() as f64
    }

    // ============================
    // PQ 생성 및 검증
    // ============================

    #[test]
    fn test_pq_new_assertions() {
        // 정상 생성
        let pq = PQIndex::new(16, 4, 256);
        assert_eq!(pq.dim(), 16);
        assert_eq!(pq.n_sub(), 4);
        assert_eq!(pq.n_sub_clusters(), 256);
        assert!(!pq.is_trained());
        assert!(pq.is_empty());

        // dim % n_sub != 0 → panic
        let result = std::panic::catch_unwind(|| PQIndex::new(15, 4, 256));
        assert!(result.is_err(), "dim=15, n_sub=4 should panic");

        // n_sub_clusters > 256 → panic
        let result = std::panic::catch_unwind(|| PQIndex::new(16, 4, 257));
        assert!(result.is_err(), "n_sub_clusters=257 should panic");

        println!("PQ new: dim={}, M={}, K'={}, trained={}, empty={}",
            pq.dim(), pq.n_sub(), pq.n_sub_clusters(), pq.is_trained(), pq.is_empty());
    }

    // ============================
    // 코드북 학습
    // ============================

    #[test]
    fn test_pq_train_codebooks() {
        let dim = 4;
        let n_sub = 2;  // 2 서브공간, 각 2차원
        let n_sub_clusters = 2;
        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);

        // 충분한 벡터로 각 서브공간에서 2개 클러스터 분리
        let mut vecs: Vec<Vec<f64>> = Vec::new();
        for i in 0..10 {
            let noise = i as f64 * 0.01;
            vecs.push(vec![0.0 + noise, 0.0 + noise, 0.0 + noise, 0.0 + noise]);
        }
        for i in 0..10 {
            let noise = i as f64 * 0.01;
            vecs.push(vec![10.0 + noise, 10.0 + noise, 10.0 + noise, 10.0 + noise]);
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        pq.train(&refs, 50, 42);
        assert!(pq.is_trained());

        // 원점 근처 벡터 → 디코딩 → 원점 근처
        let code = pq.encode(&[0.05, 0.05, 0.05, 0.05]);
        let decoded = pq.decode(&code);
        assert_eq!(decoded.len(), dim);
        for d in 0..dim {
            assert!(decoded[d].abs() < 1.0,
                "decoded[{}]={:.2} should be near 0", d, decoded[d]);
        }

        // (10,10,10,10) 근처 벡터 → 디코딩 → (10,10,10,10) 근처
        let code2 = pq.encode(&[10.0, 10.0, 10.0, 10.0]);
        let decoded2 = pq.decode(&code2);
        for d in 0..dim {
            assert!((decoded2[d] - 10.0).abs() < 1.0,
                "decoded2[{}]={:.2} should be near 10", d, decoded2[d]);
        }

        println!("train: encode([0.05..])→{:?} → decode={:?}", code, decoded);
        println!("train: encode([10.0..])→{:?} → decode={:?}", code2, decoded2);
    }

    // ============================
    // 인코딩/디코딩
    // ============================

    #[test]
    fn test_pq_encode_decode_roundtrip() {
        let dim = 8;
        let n_sub = 4;
        let n_sub_clusters = 8;
        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);

        let vecs = make_random_vectors(100, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 50, 42);

        // 훈련 벡터의 재구성 오차
        let mut total_error = 0.0;
        let mut total_norm = 0.0;
        for v in &vecs {
            let code = pq.encode(v);
            assert_eq!(code.len(), n_sub);
            let decoded = pq.decode(&code);
            assert_eq!(decoded.len(), dim);
            total_error += l2_distance(v, &decoded);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            total_norm += norm;
        }
        let relative_error = total_error / total_norm;
        println!("encode/decode: avg relative error = {:.4}", relative_error);
        assert!(relative_error < 0.5,
            "relative reconstruction error too high: {:.4}", relative_error);
    }

    // ============================
    // ADC 검색
    // ============================

    #[test]
    fn test_pq_search_basic() {
        let dim = 8;
        let n_sub = 4;
        let n_sub_clusters = 4;
        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);

        // 3개 명확히 분리된 그룹 (center = 0, 100, 200)
        let mut vecs = Vec::new();
        let mut labels = Vec::new();
        let centers = [0.0, 100.0, 200.0];
        let mut rng: u64 = 42;
        for (g, &center) in centers.iter().enumerate() {
            for _ in 0..10 {
                let v: Vec<f64> = (0..dim).map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    center + (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                }).collect();
                labels.push(format!("g{}", g));
                vecs.push(v);
            }
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        pq.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            pq.add(v, &labels[i]);
        }
        assert_eq!(pq.len(), 30);

        // 그룹 0 근처 쿼리 → top-5 모두 g0
        let query = vec![0.3; dim];
        let results = pq.search(&query, 5);
        assert_eq!(results.len(), 5);
        for (_, _, label) in &results {
            assert!(label.starts_with("g0"), "expected g0, got {}", label);
        }

        // 그룹 2 근처 쿼리 → top-5 모두 g2
        let query2 = vec![200.1; dim];
        let results2 = pq.search(&query2, 5);
        for (_, _, label) in &results2 {
            assert!(label.starts_with("g2"), "expected g2, got {}", label);
        }

        println!("search_basic: query→g0 top-5:");
        for (rank, (idx, score, label)) in results.iter().enumerate() {
            println!("  #{}: {} (idx={}, sq_l2={:.4})", rank + 1, label, idx, score);
        }
    }

    #[test]
    fn test_pq_recall_vs_bruteforce() {
        let dim = 16;
        let n_sub = 4;
        let n_sub_clusters = 16;
        let n_vecs = 500;
        let n_queries = 20;
        let k = 10;

        let vecs = make_random_vectors(n_vecs, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        let queries = make_random_vectors(n_queries, dim, 999);

        // Brute force (ground truth)
        let mut bf = BruteForceIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            bf.add(v, &format!("v{}", i));
        }

        // PQ
        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);
        pq.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            pq.add(v, &format!("v{}", i));
        }

        let mut total_recall = 0.0;
        for q in 0..n_queries {
            let exact = bf.search(&queries[q], k, Metric::L2);
            let approx = pq.search(&queries[q], k);
            total_recall += recall_at_k(&approx, &exact);
        }
        let avg_recall = total_recall / n_queries as f64;

        println!("PQ recall@{} (M={}, K'={}, N={}): {:.3}",
            k, n_sub, n_sub_clusters, n_vecs, avg_recall);
        assert!(avg_recall >= 0.2,
            "avg recall@{} should be reasonable, got {:.3}", k, avg_recall);
    }

    // ============================
    // M 파라미터 트레이드오프
    // ============================

    #[test]
    fn test_pq_m_tradeoff() {
        let dim = 16;
        let n_vecs = 300;
        let n_sub_clusters = 16;
        let k = 10;

        let vecs = make_random_vectors(n_vecs, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        let queries = make_random_vectors(10, dim, 999);

        let mut bf = BruteForceIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            bf.add(v, &format!("v{}", i));
        }

        println!("PQ M trade-off (D={}, K'={}, N={}):", dim, n_sub_clusters, n_vecs);
        for &n_sub in &[2, 4, 8, 16] {
            let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);
            pq.train(&refs, 50, 42);
            for (i, v) in vecs.iter().enumerate() {
                pq.add(v, &format!("v{}", i));
            }

            let mut total_recall = 0.0;
            for q in &queries {
                let exact = bf.search(q, k, Metric::L2);
                let approx = pq.search(q, k);
                total_recall += recall_at_k(&approx, &exact);
            }
            let avg_recall = total_recall / queries.len() as f64;
            let (orig, comp, ratio) = pq.memory_stats();
            println!("  M={:2}: recall@{}={:.3}, {:.1}x compression ({} → {} bytes)",
                n_sub, k, avg_recall, 1.0 / ratio, orig, comp);
        }
    }

    // ============================
    // 메모리 절약
    // ============================

    #[test]
    fn test_pq_memory_savings() {
        let dim = 128;
        let n_sub = 8;
        let n_sub_clusters = 256;
        let n_vecs = 1000;

        let vecs = make_random_vectors(n_vecs, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);
        pq.train(&refs, 30, 42);
        for (i, v) in vecs.iter().enumerate() {
            pq.add(v, &format!("v{}", i));
        }

        let (orig, comp, ratio) = pq.memory_stats();

        // 원본: 1000 * 128 * 8 = 1,024,000 bytes
        assert_eq!(orig, n_vecs * dim * 8);
        // 코드: 1000 * 8 = 8,000 bytes
        // 코드북: 8 * 256 * 16 * 8 = 262,144 bytes
        // 압축: 270,144 bytes
        let expected_codes = n_vecs * n_sub;
        let expected_codebook = n_sub * n_sub_clusters * (dim / n_sub) * 8;
        assert_eq!(comp, expected_codes + expected_codebook);
        assert!(ratio < 1.0, "compressed should be smaller than original");

        println!("Memory: {} → {} bytes ({:.1}x compression, ratio={:.4})",
            orig, comp, orig as f64 / comp as f64, ratio);
        println!("  codes={} bytes, codebooks={} bytes",
            expected_codes, expected_codebook);
    }

    // ============================
    // 배치 검색
    // ============================

    #[test]
    fn test_pq_batch_search() {
        let dim = 8;
        let n_sub = 4;
        let n_sub_clusters = 8;

        // 2개 그룹: center=0, center=100
        let mut vecs = Vec::new();
        for g in 0..2 {
            let center = g as f64 * 100.0;
            for j in 0..5 {
                let mut v = vec![center; dim];
                v[0] += j as f64 * 0.1;
                vecs.push(v);
            }
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);
        pq.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            pq.add(v, &format!("g{}_v{}", i / 5, i % 5));
        }

        let queries = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, dim]),
            vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 100.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        ).unwrap();

        let results = pq.batch_search(&queries, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0][0].2.starts_with("g0"), "query 0 → g0, got {}", results[0][0].2);
        assert!(results[1][0].2.starts_with("g1"), "query 1 → g1, got {}", results[1][0].2);

        println!("batch_search: query0→{}, query1→{}", results[0][0].2, results[1][0].2);
    }

    // ============================
    // 엣지 케이스
    // ============================

    #[test]
    fn test_pq_empty_and_untrained() {
        let mut pq = PQIndex::new(8, 4, 16);
        assert!(!pq.is_trained());
        assert!(pq.is_empty());

        // 학습 전 add → panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut pq2 = PQIndex::new(8, 4, 16);
            pq2.add(&[0.0; 8], "test");
        }));
        assert!(result.is_err(), "add before train should panic");

        // 학습 전 search → panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let pq2 = PQIndex::new(8, 4, 16);
            pq2.search(&[0.0; 8], 5);
        }));
        assert!(result.is_err(), "search before train should panic");

        // 학습 후 빈 상태 검색 → 빈 결과
        let vecs = make_random_vectors(20, 8, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 20, 42);
        assert!(pq.is_trained());
        assert!(pq.is_empty());

        let results = pq.search(&[0.0; 8], 5);
        assert!(results.is_empty());

        println!("empty_and_untrained: add_panic=OK, search_panic=OK, empty_search={}",
            results.len());
    }

    #[test]
    fn test_pq_get_code() {
        let dim = 8;
        let n_sub = 4;
        let n_sub_clusters = 4;
        let mut pq = PQIndex::new(dim, n_sub, n_sub_clusters);

        let vecs = make_random_vectors(50, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        pq.train(&refs, 50, 42);

        pq.add(&vecs[0], "first");
        pq.add(&vecs[0], "first_dup");  // 동일 벡터
        pq.add(&vecs[1], "second");

        // 코드 길이 = M
        assert_eq!(pq.get_code(0).len(), n_sub);
        assert_eq!(pq.get_code(1).len(), n_sub);

        // 동일 벡터 → 동일 코드
        assert_eq!(pq.get_code(0), pq.get_code(1));

        // 코드 값은 [0, K') 범위
        for &c in pq.get_code(0) {
            assert!((c as usize) < n_sub_clusters, "code {} out of range", c);
        }

        println!("get_code: v0={:?}, v0_dup={:?}, v1={:?}",
            pq.get_code(0), pq.get_code(1), pq.get_code(2));
    }
}
