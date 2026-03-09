// step72: IVF (Inverted File Index) 근사 벡터 검색
//
// Phase 4 (벡터 검색)의 두 번째 스텝.
// K-means로 벡터 공간을 K개 Voronoi cell로 분할하고 역인덱스 구축.
// 쿼리 시 nprobe개 가까운 클러스터만 탐색 → O(N·D) → O(nprobe·N/K·D).
//
// nprobe 파라미터로 속도/정확도 트레이드오프 제어:
//   nprobe=1: 가장 빠름, recall 낮음
//   nprobe=K: brute force와 동일, recall=1.0

use dezero::{
    kmeans, l2_distance,
    BruteForceIndex, IVFIndex, Metric,
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
    // K-means 테스트
    // ============================

    #[test]
    fn test_kmeans_basic() {
        // 2개의 명확히 분리된 클러스터
        let vecs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, -0.1],       // 그룹 A
            vec![10.0, 10.0], vec![10.1, 9.9], vec![9.9, 10.1],    // 그룹 B
        ];
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let (centroids, assignments) = kmeans(&refs, 2, 100, 42);

        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), 6);

        // 같은 그룹은 같은 클러스터
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        // 다른 그룹은 다른 클러스터
        assert_ne!(assignments[0], assignments[3]);

        println!("kmeans k=2: assignments={:?}", assignments);
        println!("  centroid 0: [{:.2}, {:.2}]", centroids[0][0], centroids[0][1]);
        println!("  centroid 1: [{:.2}, {:.2}]", centroids[1][0], centroids[1][1]);
    }

    #[test]
    fn test_kmeans_convergence() {
        // 중심이 클러스터 평균으로 수렴해야 함
        let vecs: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],           // 평균 (1/3, 1/3)
            vec![10.0, 10.0], vec![11.0, 10.0], vec![10.0, 11.0],     // 평균 (31/3, 31/3)
        ];
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let (centroids, assignments) = kmeans(&refs, 2, 100, 42);

        let cluster_a = assignments[0];
        let expected_a = [1.0 / 3.0, 1.0 / 3.0];
        let expected_b = [31.0 / 3.0, 31.0 / 3.0];

        let (ca, cb) = if cluster_a == 0 {
            (&centroids[0], &centroids[1])
        } else {
            (&centroids[1], &centroids[0])
        };

        for d in 0..2 {
            assert!((ca[d] - expected_a[d]).abs() < 1e-10,
                "centroid A dim {}: expected {:.4}, got {:.4}", d, expected_a[d], ca[d]);
            assert!((cb[d] - expected_b[d]).abs() < 1e-10,
                "centroid B dim {}: expected {:.4}, got {:.4}", d, expected_b[d], cb[d]);
        }

        println!("kmeans convergence: centroids = exact means");
        println!("  cluster A: ({:.4}, {:.4})", ca[0], ca[1]);
        println!("  cluster B: ({:.4}, {:.4})", cb[0], cb[1]);
    }

    // ============================
    // IVFIndex 테스트
    // ============================

    #[test]
    fn test_ivf_train_and_add() {
        let dim = 4;
        let n_clusters = 3;
        let mut ivf = IVFIndex::new(dim, n_clusters);

        assert!(!ivf.is_trained());
        assert!(ivf.is_empty());

        // 3개 클러스터 중심 근처에 벡터 생성
        let mut vecs = Vec::new();
        let centers = [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]];
        let mut rng: u64 = 123;
        for center in &centers {
            for _ in 0..10 {
                let v: Vec<f64> = center.iter().map(|&c| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    c + ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                }).collect();
                vecs.push(v);
            }
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        ivf.train(&refs, 50, 42);
        assert!(ivf.is_trained());

        // 벡터 추가
        for (i, v) in vecs.iter().enumerate() {
            ivf.add(v, &format!("vec_{}", i));
        }
        assert_eq!(ivf.len(), 30);

        let sizes = ivf.cluster_sizes();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 30);

        println!("IVF train+add: {} vectors, {} clusters, sizes={:?}", ivf.len(), n_clusters, sizes);
    }

    #[test]
    fn test_ivf_search() {
        let dim = 4;
        let n_clusters = 3;
        let mut ivf = IVFIndex::new(dim, n_clusters);

        // 잘 분리된 3개 클러스터
        let mut vecs = Vec::new();
        let mut labels = Vec::new();
        let centers = [[0.0; 4], [100.0; 4], [200.0; 4]];
        let mut rng: u64 = 77;
        for (g, center) in centers.iter().enumerate() {
            for j in 0..5 {
                let v: Vec<f64> = center.iter().map(|&c| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    c + ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0
                }).collect();
                vecs.push(v);
                labels.push(format!("g{}_v{}", g, j));
            }
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        ivf.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            ivf.add(v, &labels[i]);
        }

        // 클러스터 0 근처 쿼리 → nprobe=1이어도 정확
        let query = [0.5, 0.5, 0.5, 0.5];
        let results = ivf.search(&query, 3, 1, Metric::L2);
        assert_eq!(results.len(), 3);
        for (_, _, label) in &results {
            assert!(label.starts_with("g0"), "expected group 0, got {}", label);
        }

        println!("IVF search (nprobe=1): top-3 all from group 0");
        for (rank, (idx, score, label)) in results.iter().enumerate() {
            println!("  #{}: {} (idx={}, L2={:.4})", rank + 1, label, idx, score);
        }
    }

    #[test]
    fn test_ivf_nprobe_effect() {
        let dim = 8;
        let n_clusters = 5;
        let n_vecs = 200;

        // 랜덤 벡터 생성
        let vecs = make_random_vectors(n_vecs, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        // BruteForce (ground truth)
        let mut bf = BruteForceIndex::new(dim);
        for (i, v) in vecs.iter().enumerate() {
            bf.add(v, &format!("v{}", i));
        }

        // IVF
        let mut ivf = IVFIndex::new(dim, n_clusters);
        ivf.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            ivf.add(v, &format!("v{}", i));
        }

        // 쿼리
        let query = make_random_vectors(1, dim, 999);
        let k = 10;
        let exact = bf.search(&query[0], k, Metric::L2);

        let mut prev_recall = 0.0;
        println!("nprobe effect (K={}, N={}, k={}):", n_clusters, n_vecs, k);
        for nprobe in [1, 2, 3, n_clusters] {
            let approx = ivf.search(&query[0], k, nprobe, Metric::L2);
            let recall = recall_at_k(&approx, &exact);
            println!("  nprobe={}: recall@{}={:.2}", nprobe, k, recall);
            assert!(recall >= prev_recall - 1e-10,
                "recall should not decrease: nprobe={}, recall={:.2} < prev={:.2}", nprobe, recall, prev_recall);
            prev_recall = recall;
        }

        // nprobe=K → recall=1.0
        let full = ivf.search(&query[0], k, n_clusters, Metric::L2);
        let recall_full = recall_at_k(&full, &exact);
        assert!((recall_full - 1.0).abs() < 1e-10,
            "nprobe=K should give perfect recall, got {:.4}", recall_full);
    }

    #[test]
    fn test_ivf_recall_vs_bruteforce() {
        let dim = 16;
        let n_clusters = 10;
        let n_vecs = 500;
        let n_queries = 20;
        let k = 10;
        let nprobe = 3;

        let vecs = make_random_vectors(n_vecs, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
        let queries = make_random_vectors(n_queries, dim, 999);

        let mut bf = BruteForceIndex::new(dim);
        let mut ivf = IVFIndex::new(dim, n_clusters);
        ivf.train(&refs, 50, 42);

        for (i, v) in vecs.iter().enumerate() {
            bf.add(v, &format!("v{}", i));
            ivf.add(v, &format!("v{}", i));
        }

        let mut total_recall = 0.0;
        for q in 0..n_queries {
            let exact = bf.search(&queries[q], k, Metric::L2);
            let approx = ivf.search(&queries[q], k, nprobe, Metric::L2);
            total_recall += recall_at_k(&approx, &exact);
        }
        let avg_recall = total_recall / n_queries as f64;

        println!("IVF recall@{} (nprobe={}, K={}, N={}): {:.3}",
            k, nprobe, n_clusters, n_vecs, avg_recall);
        assert!(avg_recall >= 0.5,
            "avg recall@{} should be reasonable, got {:.3}", k, avg_recall);

        // nprobe=K → 완벽한 recall
        let mut total_full = 0.0;
        for q in 0..n_queries {
            let exact = bf.search(&queries[q], k, Metric::L2);
            let approx = ivf.search(&queries[q], k, n_clusters, Metric::L2);
            total_full += recall_at_k(&approx, &exact);
        }
        let avg_full = total_full / n_queries as f64;
        assert!((avg_full - 1.0).abs() < 1e-10, "nprobe=K should give perfect recall");
        println!("IVF recall@{} (nprobe=K={}): {:.3}", k, n_clusters, avg_full);
    }

    #[test]
    fn test_ivf_all_metrics() {
        let dim = 4;
        let n_clusters = 3;

        let vecs = make_random_vectors(30, dim, 42);
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut bf = BruteForceIndex::new(dim);
        let mut ivf = IVFIndex::new(dim, n_clusters);
        ivf.train(&refs, 50, 42);

        for (i, v) in vecs.iter().enumerate() {
            bf.add(v, &format!("v{}", i));
            ivf.add(v, &format!("v{}", i));
        }

        let query = &vecs[0]; // 첫 벡터를 쿼리로 사용
        let k = 5;

        // nprobe=K: brute force와 동일 결과
        for metric in [Metric::Cosine, Metric::DotProduct, Metric::L2, Metric::L1] {
            let exact = bf.search(query, k, metric);
            let approx = ivf.search(query, k, n_clusters, metric);
            assert_eq!(exact.len(), approx.len(), "{:?}: result count mismatch", metric);
            for i in 0..k {
                assert_eq!(exact[i].0, approx[i].0,
                    "{:?} rank {}: index {} != {}", metric, i, exact[i].0, approx[i].0);
            }
            println!("{:?}: IVF(nprobe=K) == BruteForce ✓", metric);
        }
    }

    #[test]
    fn test_ivf_batch_search() {
        let dim = 4;
        let n_clusters = 3;

        // 잘 분리된 클러스터
        let mut vecs = Vec::new();
        let centers = [[0.0; 4], [100.0; 4], [200.0; 4]];
        for (g, center) in centers.iter().enumerate() {
            for j in 0..5 {
                let mut v = center.to_vec();
                v[0] += j as f64 * 0.1;
                vecs.push(v);
            }
        }
        let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();

        let mut ivf = IVFIndex::new(dim, n_clusters);
        ivf.train(&refs, 50, 42);
        for (i, v) in vecs.iter().enumerate() {
            ivf.add(v, &format!("g{}_v{}", i / 5, i % 5));
        }

        // 2개 쿼리: 그룹 0 근처, 그룹 2 근처
        let queries = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 4]),
            vec![0.5, 0.0, 0.0, 0.0,
                 200.5, 200.0, 200.0, 200.0],
        ).unwrap();

        let results = ivf.batch_search(&queries, 2, 1, Metric::L2);
        assert_eq!(results.len(), 2);
        // 쿼리 0 → 그룹 0
        assert!(results[0][0].2.starts_with("g0"), "query 0: expected g0, got {}", results[0][0].2);
        // 쿼리 1 → 그룹 2
        assert!(results[1][0].2.starts_with("g2"), "query 1: expected g2, got {}", results[1][0].2);

        println!("batch_search: query0→{}, query1→{}", results[0][0].2, results[1][0].2);
    }

    #[test]
    fn test_ivf_empty_and_untrained() {
        let mut ivf = IVFIndex::new(4, 3);

        assert!(!ivf.is_trained());
        assert!(ivf.is_empty());
        assert_eq!(ivf.dim(), 4);
        assert_eq!(ivf.len(), 0);

        // 학습 전 add → panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ivf2 = IVFIndex::new(4, 3);
            ivf2.add(&[1.0, 2.0, 3.0, 4.0], "test");
        }));
        assert!(result.is_err(), "add before train should panic");

        // 학습 전 search → panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ivf2 = IVFIndex::new(4, 3);
            ivf2.search(&[1.0, 2.0, 3.0, 4.0], 5, 1, Metric::L2);
        }));
        assert!(result.is_err(), "search before train should panic");

        // 학습 후 빈 상태에서 검색 → 빈 결과
        let train_vecs: Vec<Vec<f64>> = vec![vec![0.0; 4], vec![1.0; 4], vec![2.0; 4]];
        let refs: Vec<&[f64]> = train_vecs.iter().map(|v| v.as_slice()).collect();
        ivf.train(&refs, 10, 42);

        assert!(ivf.is_trained());
        assert!(ivf.is_empty()); // train은 벡터를 저장하지 않음

        let results = ivf.search(&[0.5, 0.5, 0.5, 0.5], 5, 1, Metric::L2);
        assert!(results.is_empty());

        println!("edge cases: untrained panics ✓, empty search returns empty ✓");
    }
}
