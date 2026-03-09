// step77: DDPM (Denoising Diffusion Probabilistic Models)
//
// Phase 5 (생성모델)의 세 번째 스텝.
// Forward diffusion: 데이터에 점진적으로 noise 추가 → q(x_t|x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)I)
// Reverse denoising: 학습된 MLP가 noise 예측 → ε-prediction 목적함수
// Loss = E[||ε - ε_θ(x_t, t)||²]
// Sampling: x_T ~ N(0,I)부터 역방향으로 iterative denoising

use dezero::{ddpm_loss, Adam, Variable, DDPM};
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
fn test_ddpm_construction() {
    let ddpm = DDPM::new(4, 32, 50, 42);
    assert_eq!(ddpm.data_dim, 4);
    assert_eq!(ddpm.timesteps, 50);

    // forward 호출하여 lazy init
    let x = make_synthetic_data(2, 4, 99);
    let t = vec![0, 25];
    let _ = ddpm.forward(&x, &t);

    let params = ddpm.params();
    assert_eq!(params.len(), 12, "6 layers × (W + b) = 12 params");
    println!("=== test_ddpm_construction 통과 ===");
}

#[test]
fn test_noise_schedule() {
    let ddpm = DDPM::new(4, 32, 100, 42);

    // ᾱ_0 ≈ 1 (첫 스텝은 거의 noise 없음)
    let alpha_bar_0 = ddpm.alphas_cumprod[0];
    println!("  ᾱ_0 = {:.6}", alpha_bar_0);
    assert!(alpha_bar_0 > 0.999, "ᾱ_0 should be close to 1, got {}", alpha_bar_0);

    // ᾱ_{T-1} < ᾱ_0 (마지막 스텝에서 크게 감소)
    // T=100, linear β ∈ [1e-4, 0.02]일 때 ᾱ_99 ≈ 0.36
    // (T=1000이면 0에 가까워짐)
    let alpha_bar_last = ddpm.alphas_cumprod[99];
    println!("  ᾱ_99 = {:.6}", alpha_bar_last);
    assert!(alpha_bar_last < 0.5, "ᾱ_T-1 should decay significantly, got {}", alpha_bar_last);

    // 단조 감소 확인
    for t in 1..100 {
        assert!(
            ddpm.alphas_cumprod[t] < ddpm.alphas_cumprod[t - 1],
            "ᾱ should be monotonically decreasing at t={}",
            t
        );
    }

    println!("=== test_noise_schedule 통과 ===");
}

#[test]
fn test_sinusoidal_embedding() {
    let ddpm = DDPM::new(4, 32, 50, 42);
    let t = vec![0, 10, 25, 49];
    let emb = ddpm.sinusoidal_embedding(&t);

    assert_eq!(emb.shape(), vec![4, 32], "embedding shape should be [B, t_emb_dim]");

    // sin/cos 값이므로 [-1, 1] 범위
    for &v in emb.data().iter() {
        assert!(
            v >= -1.0 && v <= 1.0,
            "sinusoidal embedding should be in [-1, 1], got {}",
            v
        );
    }

    // 서로 다른 timestep은 서로 다른 embedding
    let emb_data = emb.data();
    let row0: Vec<f64> = (0..32).map(|k| emb_data[IxDyn(&[0, k])]).collect();
    let row1: Vec<f64> = (0..32).map(|k| emb_data[IxDyn(&[1, k])]).collect();
    let diff: f64 = row0.iter().zip(row1.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.1, "different timesteps should have different embeddings");

    println!("=== test_sinusoidal_embedding 통과 ===");
}

#[test]
fn test_forward_output() {
    let ddpm = DDPM::new(4, 32, 50, 42);
    let x = make_synthetic_data(8, 4, 99);
    let t = vec![5, 10, 15, 20, 25, 30, 35, 40];
    let predicted = ddpm.forward(&x, &t);

    assert_eq!(
        predicted.shape(),
        vec![8, 4],
        "predicted noise shape should match input"
    );

    // 값이 finite
    for &v in predicted.data().iter() {
        assert!(v.is_finite(), "predicted noise should be finite, got {}", v);
    }

    println!("=== test_forward_output 통과 ===");
}

#[test]
fn test_q_sample() {
    let ddpm = DDPM::new(4, 32, 100, 42);
    let x = make_synthetic_data(16, 4, 99);

    // t=0: 거의 noise 없음 → x_t ≈ x_0
    let t_early = vec![0; 16];
    let (x_t_early, _noise_early) = ddpm.q_sample(&x, &t_early);
    assert_eq!(x_t_early.shape(), vec![16, 4]);

    // x_t와 x_0의 차이가 작아야
    let diff_early: f64 = x_t_early
        .data()
        .iter()
        .zip(x.data().iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / (16 * 4) as f64;
    println!("  t=0 mean |x_t - x_0| = {:.6}", diff_early);
    assert!(diff_early < 0.5, "at t=0, x_t should be close to x_0, mean diff = {}", diff_early);

    // t=99: 거의 순수 noise → 평균 ≈ 0, 분산 ≈ 1
    let t_late = vec![99; 16];
    let (x_t_late, _noise_late) = ddpm.q_sample(&x, &t_late);
    let mean_late: f64 = x_t_late.data().iter().sum::<f64>() / (16 * 4) as f64;
    let var_late: f64 = x_t_late.data().iter().map(|v| (v - mean_late).powi(2)).sum::<f64>()
        / (16 * 4) as f64;
    println!("  t=99 mean = {:.4}, var = {:.4}", mean_late, var_late);
    // 대략적인 체크 (순수 noise에 가까움)
    assert!(mean_late.abs() < 1.5, "at t=99, mean should be near 0");

    println!("=== test_q_sample 통과 ===");
}

#[test]
fn test_ddpm_loss() {
    let ddpm = DDPM::new(4, 32, 50, 42);
    let x = make_synthetic_data(8, 4, 99);
    let t = ddpm.sample_timesteps(8);
    let loss = ddpm_loss(&ddpm, &x, &t);

    let val = loss.data()[IxDyn(&[])];
    println!("  ddpm_loss = {:.6}", val);
    assert!(val > 0.0, "loss should be positive, got {}", val);
    assert!(val.is_finite(), "loss should be finite, got {}", val);

    println!("=== test_ddpm_loss 통과 ===");
}

#[test]
fn test_gradient_flow() {
    let ddpm = DDPM::new(4, 32, 50, 42);
    let x = make_synthetic_data(8, 4, 99);
    let t = ddpm.sample_timesteps(8);

    let loss = ddpm_loss(&ddpm, &x, &t);
    ddpm.cleargrads();
    loss.backward(false, false);

    let params = ddpm.params();
    let has_grads = params.iter().filter(|p| p.grad().is_some()).count();
    println!("  params with gradient: {}/{}", has_grads, params.len());
    assert_eq!(
        has_grads,
        params.len(),
        "all {} params should have gradients, got {}",
        params.len(),
        has_grads
    );

    println!("=== test_gradient_flow 통과 ===");
}

#[test]
fn test_p_sample() {
    let ddpm = DDPM::new(4, 32, 50, 42);

    // forward를 한 번 호출하여 lazy init
    let x = make_synthetic_data(4, 4, 99);
    let t_init = vec![0; 4];
    let _ = ddpm.forward(&x, &t_init);

    // p_sample: 역방향 한 스텝
    let x_t = make_synthetic_data(4, 4, 123);
    let x_prev = ddpm.p_sample(&x_t, 25);
    assert_eq!(x_prev.shape(), vec![4, 4], "p_sample output shape");

    for &v in x_prev.data().iter() {
        assert!(v.is_finite(), "p_sample output should be finite, got {}", v);
    }

    // t=0에서는 noise 추가 없음
    let x_0_pred = ddpm.p_sample(&x_t, 0);
    assert_eq!(x_0_pred.shape(), vec![4, 4]);

    println!("=== test_p_sample 통과 ===");
}

#[test]
fn test_sampling() {
    let ddpm = DDPM::new(4, 32, 20, 42); // T=20 (빠른 테스트용)

    // forward를 한 번 호출하여 lazy init
    let x = make_synthetic_data(2, 4, 99);
    let t_init = vec![0; 2];
    let _ = ddpm.forward(&x, &t_init);

    let samples = ddpm.sample(5);
    assert_eq!(samples.shape(), vec![5, 4], "sample output shape");

    for &v in samples.data().iter() {
        assert!(v.is_finite(), "generated sample should be finite, got {}", v);
    }

    println!("=== test_sampling 통과 ===");
}

#[test]
fn test_training_convergence() {
    let data_dim = 4;
    let hidden_dim = 32;
    let timesteps = 50;
    let batch_size = 16;
    let n_samples = 64;

    let ddpm = DDPM::new(data_dim, hidden_dim, timesteps, 42);
    let optimizer = Adam::new(0.001);

    // 합성 학습 데이터 (0.5 근처 값)
    let x_all = make_synthetic_data(n_samples, data_dim, 99);
    let x_raw = x_all.data();
    let x_flat: Vec<f64> = x_raw.iter().cloned().collect();

    let mut losses = Vec::new();

    println!("\n=== DDPM 학습 ===");
    for epoch in 0..200 {
        let mut epoch_loss = 0.0;
        let mut count = 0;

        for start in (0..n_samples).step_by(batch_size) {
            let end = (start + batch_size).min(n_samples);
            let bs = end - start;
            let batch: Vec<f64> = x_flat[start * data_dim..end * data_dim].to_vec();
            let x_batch = Variable::new(
                ArrayD::from_shape_vec(IxDyn(&[bs, data_dim]), batch).unwrap(),
            );

            let t = ddpm.sample_timesteps(bs);
            let loss = ddpm_loss(&ddpm, &x_batch, &t);

            ddpm.cleargrads();
            loss.backward(false, false);
            optimizer.update(&ddpm.params());

            epoch_loss += loss.data()[IxDyn(&[])];
            count += 1;
        }

        let avg_loss = epoch_loss / count as f64;
        if epoch % 40 == 0 || epoch == 199 {
            println!("  epoch {:3}: loss = {:.6}", epoch + 1, avg_loss);
        }
        losses.push(avg_loss);
    }

    // Loss가 감소해야
    let first_loss = losses[..5].iter().sum::<f64>() / 5.0;
    let last_loss = losses[losses.len() - 5..].iter().sum::<f64>() / 5.0;
    println!("  first 5 avg = {:.6}, last 5 avg = {:.6}", first_loss, last_loss);
    assert!(
        last_loss < first_loss,
        "loss should decrease: first={:.4}, last={:.4}",
        first_loss,
        last_loss
    );

    // Loss가 finite
    assert!(last_loss.is_finite(), "final loss should be finite");

    // 샘플 생성
    let samples = ddpm.sample(5);
    assert_eq!(samples.shape(), vec![5, data_dim]);
    for &v in samples.data().iter() {
        assert!(v.is_finite(), "generated sample should be finite, got {}", v);
    }

    println!("  final loss = {:.6}", last_loss);
    println!("=== test_training_convergence 통과 ===");
}
