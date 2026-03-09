// step74: HNSW (Hierarchical Navigable Small World) 근사 벡터 검색
//
// Phase 4 (벡터 검색)의 네 번째(마지막) 스텝.
// 다층 네비게이블 스몰월드 그래프: 별도 train() 없이 벡터를 삽입하며 그래프 점진 구축.
// 검색: 최상위 레이어에서 entry point부터 greedy → layer 0에서 ef 빔서치.
//
// M과 ef_construction으로 그래프 품질 제어, ef_search로 recall/속도 트레이드오프.
// BruteForce 대비 O(N·D) → O(log N · D) 검색.

use dezero::{
    BruteForceIndex, HNSWIndex, Metric,
};
use std::collections::HashSet;
use ndarray::ArrayD;

// --- 헬퍼 ---

fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

fn recall_at_k(approx: &[(usize, f64, String)], exact: &[(usize, f64, String)]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let exact_ids: HashSet<usize> = exact.iter().map(|(i, _, _)| *i).collect();
    let hits = approx.iter().filter(|(i, _, _)| exact_ids.contains(i)).count();
    hits as f64 / exact.len() as f64
}

// --- 테스트 ---

#[test]
fn test_hnsw_new_and_params() {
    let index = HNSWIndex::new(8, 4, 16, 42);
    assert_eq!(index.dim(), 8);
    assert_eq!(index.m(), 4);
    assert_eq!(index.ef_construction(), 16);
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.entry_point(), None);

    // 빈 인덱스 검색 → 빈 결과
    let result = index.search(&[0.0; 8], 5, 10, Metric::L2);
    assert!(result.is_empty());
    println!("=== test_hnsw_new_and_params 통과 ===");
}

#[test]
fn test_hnsw_single_vector() {
    let mut index = HNSWIndex::new(4, 4, 16, 42);
    index.add(&[1.0, 2.0, 3.0, 4.0], "v0");

    assert_eq!(index.len(), 1);
    assert_eq!(index.entry_point(), Some(0));

    // 자기 자신 검색
    let result = index.search(&[1.0, 2.0, 3.0, 4.0], 1, 10, Metric::L2);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, 0);
    assert!(result[0].1 < 1e-10, "distance to self should be ~0");
    assert_eq!(result[0].2, "v0");
    println!("=== test_hnsw_single_vector 통과 ===");
}

#[test]
fn test_hnsw_basic_search() {
    // 3개 분리된 그룹: center 0, 100, 200 (dim=4)
    let mut index = HNSWIndex::new(4, 4, 32, 42);

    for i in 0..5 {
        let v = vec![i as f64 * 0.1, i as f64 * 0.1, 0.0, 0.0];
        index.add(&v, &format!("g0_{}", i));
    }
    for i in 0..5 {
        let v = vec![100.0 + i as f64 * 0.1, 100.0, 0.0, 0.0];
        index.add(&v, &format!("g1_{}", i));
    }
    for i in 0..5 {
        let v = vec![200.0 + i as f64 * 0.1, 200.0, 0.0, 0.0];
        index.add(&v, &format!("g2_{}", i));
    }

    assert_eq!(index.len(), 15);

    // 그룹 0 근처 쿼리
    let result = index.search(&[0.05, 0.05, 0.0, 0.0], 5, 15, Metric::L2);
    assert_eq!(result.len(), 5);
    for (_, _, label) in &result {
        assert!(label.starts_with("g0_"), "expected g0 group, got {}", label);
    }

    // 그룹 2 근처 쿼리
    let result = index.search(&[200.05, 200.0, 0.0, 0.0], 5, 15, Metric::L2);
    for (_, _, label) in &result {
        assert!(label.starts_with("g2_"), "expected g2 group, got {}", label);
    }
    println!("=== test_hnsw_basic_search 통과 ===");
}

#[test]
fn test_hnsw_multi_layer_structure() {
    let mut index = HNSWIndex::new(8, 4, 32, 42);

    let vectors = make_random_vectors(200, 8, 12345);
    for (i, v) in vectors.iter().enumerate() {
        index.add(v, &format!("v{}", i));
    }

    assert_eq!(index.len(), 200);

    let stats = index.graph_stats();
    println!("\n=== HNSW 그래프 통계 ===");
    for &(layer, count, avg_conn) in &stats {
        println!("  Layer {}: {} nodes, avg {:.1} connections", layer, count, avg_conn);
    }

    // Layer 0은 모든 노드 포함
    assert_eq!(stats[0].1, 200, "layer 0 should contain all nodes");

    // 상위 레이어는 더 적은 노드
    if stats.len() > 1 {
        assert!(stats[1].1 < stats[0].1, "layer 1 should have fewer nodes");
    }

    // 2개 이상의 레이어 (200노드, M=4이면 거의 확실)
    assert!(stats.len() >= 2, "should have at least 2 layers with 200 nodes");

    // layer 0 avg connections > 0
    assert!(stats[0].2 > 0.0, "layer 0 should have connections");

    println!("=== test_hnsw_multi_layer_structure 통과 ===");
}

#[test]
fn test_hnsw_recall_vs_bruteforce() {
    let n = 500;
    let dim = 16;
    let k = 10;
    let n_queries = 20;

    let vectors = make_random_vectors(n, dim, 42);
    let queries = make_random_vectors(n_queries, dim, 9999);

    // BruteForce
    let mut bf = BruteForceIndex::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        bf.add(v, &format!("v{}", i));
    }

    // HNSW
    let mut hnsw = HNSWIndex::new(dim, 16, 100, 42);
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(v, &format!("v{}", i));
    }

    let mut total_recall = 0.0;
    for q in &queries {
        let exact = bf.search(q, k, Metric::L2);
        let approx = hnsw.search(q, k, 50, Metric::L2);
        total_recall += recall_at_k(&approx, &exact);
    }
    let avg_recall = total_recall / n_queries as f64;

    println!("\n=== HNSW Recall vs BruteForce ===");
    println!("  N={}, D={}, k={}, M=16, ef_construction=100, ef_search=50", n, dim, k);
    println!("  Average recall@{}: {:.3}", k, avg_recall);

    assert!(avg_recall >= 0.7, "avg recall should be >= 0.7, got {:.3}", avg_recall);
    println!("=== test_hnsw_recall_vs_bruteforce 통과 ===");
}

#[test]
fn test_hnsw_ef_search_tradeoff() {
    let n = 500;
    let dim = 16;
    let k = 10;
    let n_queries = 20;

    let vectors = make_random_vectors(n, dim, 42);
    let queries = make_random_vectors(n_queries, dim, 7777);

    let mut bf = BruteForceIndex::new(dim);
    let mut hnsw = HNSWIndex::new(dim, 16, 100, 42);
    for (i, v) in vectors.iter().enumerate() {
        bf.add(v, &format!("v{}", i));
        hnsw.add(v, &format!("v{}", i));
    }

    let ef_values = [10, 20, 50, 100, 200];
    let mut recalls = Vec::new();

    println!("\n=== ef_search 트레이드오프 ===");
    for &ef in &ef_values {
        let mut total_recall = 0.0;
        for q in &queries {
            let exact = bf.search(q, k, Metric::L2);
            let approx = hnsw.search(q, k, ef, Metric::L2);
            total_recall += recall_at_k(&approx, &exact);
        }
        let avg = total_recall / n_queries as f64;
        println!("  ef_search={:>3}: recall@{}={:.3}", ef, k, avg);
        recalls.push(avg);
    }

    // recall은 ef_search 증가에 따라 단조 비감소 (약간의 변동 허용)
    for i in 1..recalls.len() {
        assert!(
            recalls[i] >= recalls[i - 1] - 0.05,
            "recall should generally increase with ef_search: ef={} recall={:.3} < ef={} recall={:.3}",
            ef_values[i], recalls[i], ef_values[i-1], recalls[i-1]
        );
    }

    println!("=== test_hnsw_ef_search_tradeoff 통과 ===");
}

#[test]
fn test_hnsw_all_metrics() {
    let dim = 4;
    let n = 30;
    let k = 5;

    let vectors = make_random_vectors(n, dim, 42);

    let mut bf = BruteForceIndex::new(dim);
    let mut hnsw = HNSWIndex::new(dim, 4, 32, 42);
    for (i, v) in vectors.iter().enumerate() {
        bf.add(v, &format!("v{}", i));
        hnsw.add(v, &format!("v{}", i));
    }

    let query = &vectors[0]; // 첫 벡터를 쿼리로

    println!("\n=== 전체 메트릭 테스트 ===");
    for metric in [Metric::L2, Metric::L1, Metric::Cosine, Metric::DotProduct] {
        let exact = bf.search(query, k, metric);
        // ef_search=N으로 사실상 exhaustive → exact 결과와 동일해야
        let approx = hnsw.search(query, k, n, metric);

        let recall = recall_at_k(&approx, &exact);
        println!("  {:?}: recall@{}={:.3}", metric, k, recall);

        // 작은 데이터셋 + ef=N이므로 높은 recall 기대
        assert!(recall >= 0.8, "{:?} recall should be >= 0.8, got {:.3}", metric, recall);
    }
    println!("=== test_hnsw_all_metrics 통과 ===");
}

#[test]
fn test_hnsw_batch_search() {
    let mut index = HNSWIndex::new(4, 4, 32, 42);

    // 그룹 0: (0, 0, 0, 0) 근처
    for i in 0..5 {
        index.add(&[i as f64 * 0.1, 0.0, 0.0, 0.0], &format!("g0_{}", i));
    }
    // 그룹 1: (100, 100, 100, 100) 근처
    for i in 0..5 {
        let v = 100.0 + i as f64 * 0.1;
        index.add(&[v, 100.0, 100.0, 100.0], &format!("g1_{}", i));
    }

    // 배치 쿼리: [그룹0 근처, 그룹1 근처]
    let queries = ArrayD::from_shape_vec(
        vec![2, 4],
        vec![
            0.05, 0.0, 0.0, 0.0,
            100.05, 100.0, 100.0, 100.0,
        ],
    ).unwrap();

    let results = index.batch_search(&queries, 5, 10, Metric::L2);
    assert_eq!(results.len(), 2);

    for (_, _, label) in &results[0] {
        assert!(label.starts_with("g0_"), "query 0 should find g0, got {}", label);
    }
    for (_, _, label) in &results[1] {
        assert!(label.starts_with("g1_"), "query 1 should find g1, got {}", label);
    }
    println!("=== test_hnsw_batch_search 통과 ===");
}

#[test]
fn test_hnsw_m_parameter_effect() {
    let n = 500;
    let dim = 16;
    let k = 10;
    let n_queries = 20;
    let ef_search = 50;

    let vectors = make_random_vectors(n, dim, 42);
    let queries = make_random_vectors(n_queries, dim, 5555);

    let mut bf = BruteForceIndex::new(dim);
    for (i, v) in vectors.iter().enumerate() {
        bf.add(v, &format!("v{}", i));
    }

    println!("\n=== M 파라미터 효과 ===");
    let m_values = [4, 16, 32];
    let mut recalls = Vec::new();

    for &m in &m_values {
        let mut hnsw = HNSWIndex::new(dim, m, 100, 42);
        for (i, v) in vectors.iter().enumerate() {
            hnsw.add(v, &format!("v{}", i));
        }

        let stats = hnsw.graph_stats();
        let layers = stats.len();
        let avg_conn_l0 = stats[0].2;

        let mut total_recall = 0.0;
        for q in &queries {
            let exact = bf.search(q, k, Metric::L2);
            let approx = hnsw.search(q, k, ef_search, Metric::L2);
            total_recall += recall_at_k(&approx, &exact);
        }
        let avg = total_recall / n_queries as f64;
        println!("  M={:>2}: recall@{}={:.3}, layers={}, avg_conn_L0={:.1}",
                 m, k, avg, layers, avg_conn_l0);
        recalls.push(avg);
    }

    // M이 클수록 recall 비슷하거나 더 높아야 (약간의 변동 허용)
    assert!(
        recalls[2] >= recalls[0] - 0.1,
        "M=32 recall ({:.3}) should be >= M=4 recall ({:.3}) - 0.1",
        recalls[2], recalls[0]
    );

    println!("=== test_hnsw_m_parameter_effect 통과 ===");
}

#[test]
fn test_hnsw_deterministic_with_seed() {
    let dim = 8;
    let vectors = make_random_vectors(50, dim, 42);

    // 동일 seed → 동일 결과
    let mut idx1 = HNSWIndex::new(dim, 4, 16, 12345);
    let mut idx2 = HNSWIndex::new(dim, 4, 16, 12345);
    for (i, v) in vectors.iter().enumerate() {
        idx1.add(v, &format!("v{}", i));
        idx2.add(v, &format!("v{}", i));
    }

    let query = &vectors[0];
    let r1 = idx1.search(query, 5, 20, Metric::L2);
    let r2 = idx2.search(query, 5, 20, Metric::L2);

    assert_eq!(r1.len(), r2.len());
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.0, b.0, "same seed should give same results");
        assert!((a.1 - b.1).abs() < 1e-10);
    }

    // 다른 seed → 다를 수 있음 (max_level이 다를 가능성)
    let mut idx3 = HNSWIndex::new(dim, 4, 16, 99999);
    for (i, v) in vectors.iter().enumerate() {
        idx3.add(v, &format!("v{}", i));
    }
    // 그래프 구조가 다를 수 있으나, 검색 결과는 여전히 유효해야
    let r3 = idx3.search(query, 5, 20, Metric::L2);
    assert_eq!(r3.len(), 5);

    println!("=== test_hnsw_deterministic_with_seed 통과 ===");
}

#[test]
fn test_hnsw_graph_stats_detail() {
    let mut index = HNSWIndex::new(4, 8, 32, 42);

    // 빈 인덱스
    assert!(index.graph_stats().is_empty());

    // 100개 삽입
    let vectors = make_random_vectors(100, 4, 42);
    for (i, v) in vectors.iter().enumerate() {
        index.add(v, &format!("v{}", i));
    }

    let stats = index.graph_stats();
    assert!(!stats.is_empty());

    // 각 레이어의 노드 수 합계 출력
    println!("\n=== 상세 그래프 통계 ===");
    let mut prev_count = usize::MAX;
    for &(layer, count, avg_conn) in &stats {
        println!("  Layer {}: {} nodes, avg {:.2} connections", layer, count, avg_conn);
        // 상위 레이어일수록 노드 수 감소 (비증가)
        assert!(count <= prev_count, "layer {} has more nodes than layer below", layer);
        prev_count = count;
        // 연결 수는 양수 (빈 레이어 제외)
        if count > 1 {
            assert!(avg_conn > 0.0, "layer {} should have connections", layer);
        }
    }

    // max_level과 stats 길이 일치
    assert_eq!(stats.len(), index.max_level() + 1);

    println!("=== test_hnsw_graph_stats_detail 통과 ===");
}
