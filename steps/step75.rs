// step75: VAE (Variational Autoencoder)
//
// Phase 5 (생성모델)의 첫 번째 스텝.
// Encoder → Latent Space → Decoder 구조.
// Reparameterization trick으로 역전파 가능한 샘플링.
// ELBO = Reconstruction Loss + KL Divergence 최소화.

use dezero::{vae_loss, Adam, Variable, VAE};
use ndarray::{ArrayD, IxDyn};

// --- 헬퍼 ---

/// LCG 기반 [0,1) 합성 데이터 생성
fn make_synthetic_data(n: usize, dim: usize, seed: u64) -> Variable {
    let mut rng = seed;
    let data: Vec<f64> = (0..n * dim)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect();
    Variable::new(ArrayD::from_shape_vec(IxDyn(&[n, dim]), data).unwrap())
}

// --- 테스트 ---

#[test]
fn test_vae_construction() {
    let vae = VAE::new(784, 256, 20, 1.0, 42);
    assert_eq!(vae.latent_dim, 20);
    assert!((vae.beta - 1.0).abs() < 1e-10);

    // 아직 forward 전이므로 params는 b만 5개 (W는 lazy init)
    // forward 후에는 W 5개 + b 5개 = 10개
    let x = make_synthetic_data(2, 784, 99);
    let _ = vae.forward(&x);
    let params = vae.params();
    assert_eq!(params.len(), 10, "5 Linear layers × (W + b) = 10 params");
    println!("=== test_vae_construction 통과 ===");
}

#[test]
fn test_encode_shape() {
    let vae = VAE::new(32, 16, 4, 1.0, 42);
    let x = make_synthetic_data(8, 32, 99);

    let (mu, log_var) = vae.encode(&x);
    assert_eq!(mu.shape(), vec![8, 4], "mu shape should be (B, latent_dim)");
    assert_eq!(log_var.shape(), vec![8, 4], "log_var shape should be (B, latent_dim)");
    println!("=== test_encode_shape 통과 ===");
}

#[test]
fn test_decode_shape_and_range() {
    let vae = VAE::new(32, 16, 4, 1.0, 42);
    // decode에 먼저 encode를 통해 latent_dim 초기화
    let x = make_synthetic_data(8, 32, 99);
    let _ = vae.forward(&x);

    // 직접 z 생성하여 decode
    let z = make_synthetic_data(5, 4, 77);
    let x_recon = vae.decode(&z);

    assert_eq!(x_recon.shape(), vec![5, 32], "decode output shape should be (B, input_dim)");

    // sigmoid 출력이므로 모든 값이 (0, 1) 범위
    let data = x_recon.data();
    for &v in data.iter() {
        assert!(v > 0.0 && v < 1.0, "sigmoid output should be in (0,1), got {}", v);
    }
    println!("=== test_decode_shape_and_range 통과 ===");
}

#[test]
fn test_forward_full() {
    let vae = VAE::new(32, 16, 4, 1.0, 42);
    let x = make_synthetic_data(8, 32, 99);

    let (x_recon, mu, log_var) = vae.forward(&x);

    assert_eq!(x_recon.shape(), vec![8, 32], "x_recon shape");
    assert_eq!(mu.shape(), vec![8, 4], "mu shape");
    assert_eq!(log_var.shape(), vec![8, 4], "log_var shape");

    // x_recon은 sigmoid 출력
    for &v in x_recon.data().iter() {
        assert!(v > 0.0 && v < 1.0, "x_recon should be in (0,1)");
    }
    println!("=== test_forward_full 통과 ===");
}

#[test]
fn test_reparameterization_gradient() {
    // reparameterization trick의 핵심: mu와 log_var에 gradient가 흘러야 함
    let vae = VAE::new(16, 8, 2, 1.0, 42);
    let x = make_synthetic_data(4, 16, 99);

    let (x_recon, mu, log_var) = vae.forward(&x);
    let (total, _, _) = vae_loss(&x, &x_recon, &mu, &log_var, 1.0);

    vae.cleargrads();
    total.backward(false, false);

    // encoder의 W, b에 gradient가 존재해야
    let params = vae.params();
    let has_grads = params.iter().filter(|p| p.grad().is_some()).count();
    assert!(has_grads >= 8, "at least 8 params should have gradients, got {}", has_grads);
    println!("  gradient가 있는 파라미터: {}/{}", has_grads, params.len());
    println!("=== test_reparameterization_gradient 통과 ===");
}

#[test]
fn test_kl_standard_normal() {
    // mu=0, log_var=0 (즉 sigma=1) → q(z|x) = p(z) = N(0,1) → KL = 0
    let batch = 4;
    let latent = 3;
    let mu = Variable::new(ArrayD::zeros(IxDyn(&[batch, latent])));
    let log_var = Variable::new(ArrayD::zeros(IxDyn(&[batch, latent])));
    let x = Variable::new(ArrayD::zeros(IxDyn(&[batch, 10])));
    let x_recon = Variable::new(ArrayD::zeros(IxDyn(&[batch, 10])));

    let (_, _, kl) = vae_loss(&x, &x_recon, &mu, &log_var, 1.0);
    let kl_val = kl.data()[ndarray::IxDyn(&[])];

    println!("  KL(N(0,1) || N(0,1)) = {:.6}", kl_val);
    assert!(kl_val.abs() < 1e-10, "KL should be ~0 for standard normal, got {}", kl_val);
    println!("=== test_kl_standard_normal 통과 ===");
}

#[test]
fn test_vae_loss_decomposition() {
    let vae = VAE::new(16, 8, 2, 1.0, 42);
    let x = make_synthetic_data(4, 16, 99);
    let (x_recon, mu, log_var) = vae.forward(&x);

    let beta = 2.5;
    let (total, recon, kl) = vae_loss(&x, &x_recon, &mu, &log_var, beta);

    let total_val = total.data()[ndarray::IxDyn(&[])];
    let recon_val = recon.data()[ndarray::IxDyn(&[])];
    let kl_val = kl.data()[ndarray::IxDyn(&[])];

    let expected = recon_val + beta * kl_val;
    let diff = (total_val - expected).abs();

    println!("  total={:.6}, recon={:.6}, kl={:.6}, beta={}", total_val, recon_val, kl_val, beta);
    println!("  recon + beta*kl = {:.6}, diff = {:.2e}", expected, diff);
    assert!(diff < 1e-8, "total should equal recon + beta*kl, diff={:.2e}", diff);
    println!("=== test_vae_loss_decomposition 통과 ===");
}

#[test]
fn test_training_loss_decreases() {
    let input_dim = 16;
    let hidden_dim = 32;
    let latent_dim = 4;
    let batch_size = 16;
    let n_samples = 64;

    let vae = VAE::new(input_dim, hidden_dim, latent_dim, 1.0, 42);
    let optimizer = Adam::new(0.001);
    let x_all = make_synthetic_data(n_samples, input_dim, 99);
    let x_raw = x_all.data();
    let x_flat: Vec<f64> = x_raw.iter().cloned().collect();

    let mut losses = Vec::new();

    println!("\n=== VAE 학습 ===");
    for epoch in 0..30 {
        let mut epoch_loss = 0.0;
        let mut count = 0;

        for start in (0..n_samples).step_by(batch_size) {
            let end = (start + batch_size).min(n_samples);
            let bs = end - start;
            let batch: Vec<f64> = x_flat[start * input_dim..end * input_dim].to_vec();
            let x_batch = Variable::new(
                ArrayD::from_shape_vec(IxDyn(&[bs, input_dim]), batch).unwrap(),
            );

            let (x_recon, mu, log_var) = vae.forward(&x_batch);
            let (total, _, _) = vae_loss(&x_batch, &x_recon, &mu, &log_var, vae.beta);

            vae.cleargrads();
            total.backward(false, false);
            optimizer.update(&vae.params());

            epoch_loss += total.data()[ndarray::IxDyn(&[])];
            count += 1;
        }

        let avg = epoch_loss / count as f64;
        if epoch % 10 == 0 || epoch == 29 {
            println!("  epoch {:2}: loss = {:.4}", epoch + 1, avg);
        }
        losses.push(avg);
    }

    // loss가 감소해야
    let first = losses[0];
    let last = *losses.last().unwrap();
    println!("  first_loss={:.4}, last_loss={:.4}", first, last);
    assert!(last < first, "loss should decrease: first={:.4} > last={:.4}", first, last);
    println!("=== test_training_loss_decreases 통과 ===");
}

#[test]
fn test_sample_generation() {
    let vae = VAE::new(16, 8, 2, 1.0, 42);
    // forward 한번 호출하여 decoder 가중치 초기화
    let x = make_synthetic_data(2, 16, 99);
    let _ = vae.forward(&x);

    let samples = vae.sample(10);
    assert_eq!(samples.shape(), vec![10, 16], "sample shape should be (num_samples, input_dim)");

    // sigmoid 출력이므로 (0,1)
    for &v in samples.data().iter() {
        assert!(v > 0.0 && v < 1.0, "sample should be in (0,1), got {}", v);
    }
    println!("=== test_sample_generation 통과 ===");
}

#[test]
fn test_beta_effect() {
    let input_dim = 16;
    let hidden_dim = 16;
    let latent_dim = 2;
    let x = make_synthetic_data(8, input_dim, 99);

    println!("\n=== beta 효과 ===");
    for &beta in &[0.01, 1.0, 10.0] {
        let vae = VAE::new(input_dim, hidden_dim, latent_dim, beta, 42);
        let (x_recon, mu, log_var) = vae.forward(&x);
        let (total, recon, kl) = vae_loss(&x, &x_recon, &mu, &log_var, beta);

        let total_val = total.data()[ndarray::IxDyn(&[])];
        let recon_val = recon.data()[ndarray::IxDyn(&[])];
        let kl_val = kl.data()[ndarray::IxDyn(&[])];

        println!("  beta={:>5.2}: total={:.4}, recon={:.4}, kl={:.4}", beta, total_val, recon_val, kl_val);
    }
    // beta가 다르면 loss 균형이 달라짐 (정량적 assert는 학습 후에만 의미)
    println!("=== test_beta_effect 통과 ===");
}

#[test]
fn test_cleargrads() {
    let vae = VAE::new(16, 8, 2, 1.0, 42);
    let x = make_synthetic_data(4, 16, 99);

    let (x_recon, mu, log_var) = vae.forward(&x);
    let (total, _, _) = vae_loss(&x, &x_recon, &mu, &log_var, 1.0);
    total.backward(false, false);

    // backward 후 gradient 존재 확인
    let params = vae.params();
    let has_grads_before = params.iter().filter(|p| p.grad().is_some()).count();
    assert!(has_grads_before > 0, "should have gradients after backward");

    // cleargrads 후 gradient 없어야
    vae.cleargrads();
    let has_grads_after = params.iter().filter(|p| p.grad().is_some()).count();
    assert_eq!(has_grads_after, 0, "should have no gradients after cleargrads");
    println!("=== test_cleargrads 통과 ===");
}
