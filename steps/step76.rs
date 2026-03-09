// step76: GAN (Generative Adversarial Network)
//
// Phase 5 (생성모델)의 두 번째 스텝.
// Generator vs Discriminator의 적대적 학습 (minimax game).
// Non-saturating G loss로 학습 초기 gradient 문제 해결.
// D loss = bce(D(real), 1) + bce(D(fake), 0)
// G loss = bce(D(G(z)), 1)

use dezero::{binary_cross_entropy, gan_loss, Adam, Variable, GAN};
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
fn test_gan_construction() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    assert_eq!(gan.latent_dim, 4);
    assert_eq!(gan.data_dim, 8);

    // forward 호출하여 lazy init
    let x = make_synthetic_data(2, 8, 99);
    let _ = gan.discriminate(&x);
    let _ = gan.generate(2);

    let g_params = gan.generator_params();
    let d_params = gan.discriminator_params();
    assert_eq!(g_params.len(), 6, "G: 3 layers × (W + b) = 6 params");
    assert_eq!(d_params.len(), 6, "D: 3 layers × (W + b) = 6 params");
    assert_eq!(gan.params().len(), 12, "total params = 12");
    println!("=== test_gan_construction 통과 ===");
}

#[test]
fn test_generator_output() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    let fake = gan.generate(10);

    assert_eq!(fake.shape(), vec![10, 8], "generator output shape");

    // sigmoid 출력이므로 (0, 1)
    for &v in fake.data().iter() {
        assert!(v > 0.0 && v < 1.0, "generator output should be in (0,1), got {}", v);
    }
    println!("=== test_generator_output 통과 ===");
}

#[test]
fn test_discriminator_output() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    let x = make_synthetic_data(6, 8, 99);
    let d_out = gan.discriminate(&x);

    assert_eq!(d_out.shape(), vec![6, 1], "discriminator output shape (B, 1)");

    // sigmoid 출력이므로 (0, 1)
    for &v in d_out.data().iter() {
        assert!(v > 0.0 && v < 1.0, "discriminator output should be in (0,1), got {}", v);
    }
    println!("=== test_discriminator_output 통과 ===");
}

#[test]
fn test_bce_known_values() {
    // Case 1: p=0.8, t=1.0 → -log(0.8) ≈ 0.22314
    let p1 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.8]).unwrap());
    let t1 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap());
    let loss1 = binary_cross_entropy(&p1, &t1);
    let expected1 = -(0.8_f64.ln());
    let val1 = loss1.data()[IxDyn(&[])];
    println!("  bce(0.8, 1.0) = {:.6}, expected = {:.6}", val1, expected1);
    assert!((val1 - expected1).abs() < 1e-5, "got {}", val1);

    // Case 2: p=0.3, t=0.0 → -log(1-0.3) = -log(0.7) ≈ 0.35667
    let p2 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.3]).unwrap());
    let t2 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap());
    let loss2 = binary_cross_entropy(&p2, &t2);
    let expected2 = -(0.7_f64.ln());
    let val2 = loss2.data()[IxDyn(&[])];
    println!("  bce(0.3, 0.0) = {:.6}, expected = {:.6}", val2, expected2);
    assert!((val2 - expected2).abs() < 1e-5, "got {}", val2);

    // Case 3: 배치 — 평균 확인
    let p3 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.8, 0.3]).unwrap());
    let t3 = Variable::new(ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 0.0]).unwrap());
    let loss3 = binary_cross_entropy(&p3, &t3);
    let expected3 = (expected1 + expected2) / 2.0;
    let val3 = loss3.data()[IxDyn(&[])];
    println!("  bce([0.8,0.3], [1,0]) = {:.6}, expected = {:.6}", val3, expected3);
    assert!((val3 - expected3).abs() < 1e-5, "got {}", val3);

    println!("=== test_bce_known_values 통과 ===");
}

#[test]
fn test_bce_gradient() {
    let p = Variable::new(ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![0.2, 0.5, 0.8, 0.9]).unwrap());
    let t = Variable::new(ArrayD::from_shape_vec(IxDyn(&[4, 1]), vec![0.0, 1.0, 1.0, 0.0]).unwrap());
    let loss = binary_cross_entropy(&p, &t);
    loss.backward(false, false);

    let grad = p.grad();
    assert!(grad.is_some(), "p should have gradient after backward");
    let g = grad.unwrap();
    assert_eq!(g.shape(), &[4, 1], "gradient shape should match p");
    println!("  gradient = {:?}", g);
    println!("=== test_bce_gradient 통과 ===");
}

#[test]
fn test_d_loss_decomposition() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    let x_real = make_synthetic_data(4, 8, 99);
    let d_real = gan.discriminate(&x_real);
    let fake = gan.generate(4);
    let d_fake = gan.discriminate(&fake);

    let (d_loss, _g_loss) = gan_loss(&d_real, &d_fake);

    // 수동 계산
    let ones = Variable::new(ArrayD::ones(IxDyn(&[4, 1])));
    let zeros = Variable::new(ArrayD::zeros(IxDyn(&[4, 1])));
    let d_loss_manual = &binary_cross_entropy(&d_real, &ones) + &binary_cross_entropy(&d_fake, &zeros);

    let val = d_loss.data()[IxDyn(&[])];
    let val_manual = d_loss_manual.data()[IxDyn(&[])];
    let diff = (val - val_manual).abs();

    println!("  d_loss = {:.6}, manual = {:.6}, diff = {:.2e}", val, val_manual, diff);
    assert!(diff < 1e-10, "d_loss should equal manual computation");
    assert!(val > 0.0 && val.is_finite(), "d_loss should be positive and finite");
    println!("=== test_d_loss_decomposition 통과 ===");
}

#[test]
fn test_g_loss_computation() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    let fake = gan.generate(4);
    let d_fake = gan.discriminate(&fake);

    // g_loss = bce(d_fake, ones)
    let ones = Variable::new(ArrayD::ones(IxDyn(&[4, 1])));
    let g_loss = binary_cross_entropy(&d_fake, &ones);

    let val = g_loss.data()[IxDyn(&[])];
    println!("  g_loss = {:.6}", val);
    assert!(val > 0.0 && val.is_finite(), "g_loss should be positive and finite, got {}", val);
    println!("=== test_g_loss_computation 통과 ===");
}

#[test]
fn test_gradient_flow_discriminator() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    let x_real = make_synthetic_data(4, 8, 99);
    let d_real = gan.discriminate(&x_real);
    let fake = gan.generate(4);
    let d_fake = gan.discriminate(&fake);

    let (d_loss, _) = gan_loss(&d_real, &d_fake);
    gan.cleargrads();
    d_loss.backward(false, false);

    let d_params = gan.discriminator_params();
    let has_grads = d_params.iter().filter(|p| p.grad().is_some()).count();
    println!("  D params with gradient: {}/{}", has_grads, d_params.len());
    assert!(has_grads >= 5, "at least 5 D params should have gradients, got {}", has_grads);
    println!("=== test_gradient_flow_discriminator 통과 ===");
}

#[test]
fn test_gradient_flow_generator() {
    let gan = GAN::new(8, 4, 16, 16, 42);
    // G forward → D forward → g_loss → backward
    let fake = gan.generate(4);
    let d_fake = gan.discriminate(&fake);
    let ones = Variable::new(ArrayD::ones(IxDyn(&[4, 1])));
    let g_loss = binary_cross_entropy(&d_fake, &ones);

    gan.cleargrads();
    g_loss.backward(false, false);

    // G의 gradient가 D를 통과하여 G까지 흘러야 함
    let g_params = gan.generator_params();
    let has_grads = g_params.iter().filter(|p| p.grad().is_some()).count();
    println!("  G params with gradient: {}/{}", has_grads, g_params.len());
    assert!(has_grads >= 5, "at least 5 G params should have gradients, got {}", has_grads);
    println!("=== test_gradient_flow_generator 통과 ===");
}

#[test]
fn test_training_convergence() {
    let data_dim = 4;
    let latent_dim = 2;
    let g_hidden = 16;
    let d_hidden = 16;
    let batch_size = 16;
    let n_samples = 64;

    let gan = GAN::new(data_dim, latent_dim, g_hidden, d_hidden, 42);
    let d_optimizer = Adam::new(0.0002);
    let g_optimizer = Adam::new(0.0002);

    // 합성 학습 데이터
    let x_all = make_synthetic_data(n_samples, data_dim, 99);
    let x_raw = x_all.data();
    let x_flat: Vec<f64> = x_raw.iter().cloned().collect();

    let mut d_losses = Vec::new();
    let mut g_losses = Vec::new();

    println!("\n=== GAN 학습 ===");
    for epoch in 0..100 {
        let mut epoch_d_loss = 0.0;
        let mut epoch_g_loss = 0.0;
        let mut count = 0;

        for start in (0..n_samples).step_by(batch_size) {
            let end = (start + batch_size).min(n_samples);
            let bs = end - start;
            let batch: Vec<f64> = x_flat[start * data_dim..end * data_dim].to_vec();
            let x_batch = Variable::new(
                ArrayD::from_shape_vec(IxDyn(&[bs, data_dim]), batch).unwrap(),
            );

            // --- D 학습 ---
            let d_real = gan.discriminate(&x_batch);
            let fake = gan.generate(bs);
            let d_fake = gan.discriminate(&fake);
            let (d_loss, _) = gan_loss(&d_real, &d_fake);

            gan.cleargrads();
            d_loss.backward(false, false);
            d_optimizer.update(&gan.discriminator_params());

            // --- G 학습 ---
            let fake2 = gan.generate(bs);
            let d_fake2 = gan.discriminate(&fake2);
            let ones = Variable::new(ArrayD::ones(IxDyn(&[bs, 1])));
            let g_loss = binary_cross_entropy(&d_fake2, &ones);

            gan.cleargrads();
            g_loss.backward(false, false);
            g_optimizer.update(&gan.generator_params());

            epoch_d_loss += d_loss.data()[IxDyn(&[])];
            epoch_g_loss += g_loss.data()[IxDyn(&[])];
            count += 1;
        }

        let avg_d = epoch_d_loss / count as f64;
        let avg_g = epoch_g_loss / count as f64;

        if epoch % 20 == 0 || epoch == 99 {
            println!("  epoch {:3}: d_loss = {:.4}, g_loss = {:.4}", epoch + 1, avg_d, avg_g);
        }
        d_losses.push(avg_d);
        g_losses.push(avg_g);
    }

    // D loss가 발산하지 않아야 (bounded)
    let last_d = *d_losses.last().unwrap();
    assert!(last_d.is_finite(), "d_loss should be finite, got {}", last_d);
    assert!(last_d < 10.0, "d_loss should not explode, got {:.4}", last_d);

    // G loss가 유한해야
    let last_g = *g_losses.last().unwrap();
    assert!(last_g.is_finite(), "g_loss should be finite, got {}", last_g);

    // 생성된 샘플 확인
    let samples = gan.generate(5);
    assert_eq!(samples.shape(), vec![5, data_dim]);
    for &v in samples.data().iter() {
        assert!(v > 0.0 && v < 1.0, "generated sample should be in (0,1), got {}", v);
    }

    println!("  final d_loss={:.4}, g_loss={:.4}", last_d, last_g);
    println!("=== test_training_convergence 통과 ===");
}
