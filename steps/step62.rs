// step62: Transpose(axes)와 Batched Matmul
//
// Attention 계산에 필요한 텐서 조작 연산
//
// Transpose(axes):
//   기존 transpose는 2D 전치 (M,N) → (N,M) 만 지원
//   Attention에서는 임의 축 순열이 필요:
//     (B, H, T, D) → (B, T, H, D)  ← axes = [0, 2, 1, 3]
//   역전파: 역순열(inverse permutation)을 적용
//     axes = [0, 2, 1, 3] → inv = [0, 2, 1, 3] (이 경우 자기 역)
//     일반: inv_axes[axes[i]] = i
//
// Batched Matmul:
//   기존 matmul은 2D (M,K) @ (K,N) 전용
//   Attention: (B, H, T, D) @ (B, H, D, T) → (B, H, T, T)
//   배치 차원(앞쪽)을 유지한 채 마지막 2차원에서 행렬곱
//   역전파:
//     gx = gy @ w^T  (마지막 두 차원 전치)
//     gw = x^T @ gy

use dezero::{transpose_axes, batched_matmul, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_axes_3d() {
        // (2, 3, 4) → axes [0, 2, 1] → (2, 4, 3)
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 4]), data).unwrap(),
        );

        let y = transpose_axes(&x, &[0, 2, 1]);
        assert_eq!(y.shape(), vec![2, 4, 3]);

        // 값 검증: x[0,1,2] = 1*4+2 = 6 → y[0,2,1] = 6
        let y_data = y.data();
        assert_eq!(y_data[[0, 2, 1]], 6.0);
        println!("3D transpose shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_transpose_axes_4d() {
        // (2, 3, 4, 5) → axes [0, 2, 1, 3] → (2, 4, 3, 5)
        // Attention에서 (B, H, T, D) → (B, T, H, D)와 동일한 패턴
        let data: Vec<f64> = (0..120).map(|i| i as f64).collect();
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3, 4, 5]), data).unwrap(),
        );

        let y = transpose_axes(&x, &[0, 2, 1, 3]);
        assert_eq!(y.shape(), vec![2, 4, 3, 5]);

        // x[1,2,3,4] → y[1,3,2,4]
        let x_data = x.data();
        let y_data = y.data();
        assert_eq!(x_data[[1, 2, 3, 4]], y_data[[1, 3, 2, 4]]);
        println!("4D transpose shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_transpose_axes_backward() {
        // backward: 역순열이 적용되어 원래 shape로 복원
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3, 4]),
                (0..24).map(|i| i as f64).collect(),
            ).unwrap(),
        );

        let y = transpose_axes(&x, &[0, 2, 1]); // (2,3,4) → (2,4,3)
        let loss = dezero::sum(&y);
        loss.backward(false, false);

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), &[2, 3, 4]);
        // sum의 기울기는 모두 1, transpose를 통과하면 shape만 복원됨
        assert!(grad.iter().all(|&v| (v - 1.0).abs() < 1e-10));
        println!("transpose_axes backward shape: {:?} ✓", grad.shape());
    }

    #[test]
    fn test_batched_matmul_3d() {
        // (2, 3, 4) @ (2, 4, 5) → (2, 3, 5)
        // 배치=2, (3,4) @ (4,5) = (3,5)
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3, 4]),
                (0..24).map(|i| i as f64).collect(),
            ).unwrap(),
        );
        let w = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 4, 5]),
                (0..40).map(|i| i as f64 * 0.1).collect(),
            ).unwrap(),
        );

        let y = batched_matmul(&x, &w);
        assert_eq!(y.shape(), vec![2, 3, 5]);

        // 수동 검증: y[0,0,:] = x[0,0,:] @ w[0,:,:]
        // x[0,0,:] = [0,1,2,3], w[0,:,:] = [[0,0.1,...,0.4],[0.5,...,0.9],...]
        let y_data = y.data();
        println!("batched matmul [0,0,0] = {:.4}", y_data[[0, 0, 0]]);
        println!("3D batched matmul shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_batched_matmul_4d() {
        // Attention 패턴: (B, H, T, D) @ (B, H, D, T) → (B, H, T, T)
        // B=2, H=4, T=3, D=5
        let x_data: Vec<f64> = (0..120).map(|i| i as f64 * 0.01).collect();
        let w_data: Vec<f64> = (0..120).map(|i| i as f64 * 0.01).collect();

        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 4, 3, 5]), x_data).unwrap(),
        );
        let w = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 4, 5, 3]), w_data).unwrap(),
        );

        let y = batched_matmul(&x, &w);
        assert_eq!(y.shape(), vec![2, 4, 3, 3]);
        println!("4D batched matmul (attention pattern) shape: {:?} ✓", y.shape());
    }

    #[test]
    fn test_batched_matmul_backward() {
        // backward 테스트: 기울기 shape 확인
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3, 4]),
                (0..24).map(|i| i as f64 * 0.1).collect(),
            ).unwrap(),
        );
        let w = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 4, 5]),
                (0..40).map(|i| i as f64 * 0.1).collect(),
            ).unwrap(),
        );

        let y = batched_matmul(&x, &w);
        let loss = dezero::sum(&y);
        loss.backward(false, false);

        let gx = x.grad().unwrap();
        let gw = w.grad().unwrap();
        assert_eq!(gx.shape(), &[2, 3, 4]);
        assert_eq!(gw.shape(), &[2, 4, 5]);
        println!("batched matmul backward gx shape: {:?} ✓", gx.shape());
        println!("batched matmul backward gw shape: {:?} ✓", gw.shape());
    }

    #[test]
    fn test_attention_pattern() {
        // 실제 Attention 시뮬레이션:
        // Q, K, V: (B=1, H=2, T=4, D=3)
        // scores = Q @ K^T / sqrt(D) → (1, 2, 4, 4)
        let q_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1).collect();
        let k_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1 + 0.05).collect();
        let v_data: Vec<f64> = (0..24).map(|i| i as f64 * 0.1).collect();

        let q = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), q_data).unwrap(),
        );
        let k = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), k_data).unwrap(),
        );
        let v = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 4, 3]), v_data).unwrap(),
        );

        // K^T: (1,2,4,3) → (1,2,3,4)
        let k_t = transpose_axes(&k, &[0, 1, 3, 2]);
        assert_eq!(k_t.shape(), vec![1, 2, 3, 4]);

        // scores = Q @ K^T → (1, 2, 4, 4)
        let scores = batched_matmul(&q, &k_t);
        assert_eq!(scores.shape(), vec![1, 2, 4, 4]);

        // scaled scores
        let d_k = 3.0_f64;
        let scaled = &scores / d_k.sqrt();
        assert_eq!(scaled.shape(), vec![1, 2, 4, 4]);

        // attention output = softmax(scaled) @ V → (1, 2, 4, 3)
        // (softmax 아직 미구현이므로 바로 V와 곱)
        let out = batched_matmul(&scaled, &v);
        assert_eq!(out.shape(), vec![1, 2, 4, 3]);

        // backward
        let loss = dezero::sum(&out);
        loss.backward(false, false);

        assert_eq!(q.grad().unwrap().shape(), &[1, 2, 4, 3]);
        assert_eq!(k.grad().unwrap().shape(), &[1, 2, 4, 3]);
        assert_eq!(v.grad().unwrap().shape(), &[1, 2, 4, 3]);

        println!("Attention pattern: Q@K^T → scores → scores@V");
        println!("  Q shape:      {:?}", q.shape());
        println!("  K^T shape:    {:?}", k_t.shape());
        println!("  scores shape: {:?}", scores.shape());
        println!("  output shape: {:?}", out.shape());
        println!("  Q grad shape: {:?}", q.grad().unwrap().shape());
        println!("Attention pattern test passed! ✓");
    }
}
