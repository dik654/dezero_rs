// step53: 모델 가중치 저장과 로드
//
// step51과의 차이:
//   step51: MNIST 학습 후 프로그램 종료 → 학습된 가중치 소실
//   step53: 학습된 가중치를 파일로 저장 → 다음 실행 시 이어서 학습 가능
//
// Python DeZero:
//   model.save_weights('my_mlp.npz')  — NumPy의 .npz 형식 (zip of .npy)
//   model.load_weights('my_mlp.npz')  — .npz에서 파라미터 복원
//
// Rust 구현:
//   model.save_weights("my_mlp.bin")  — 커스텀 바이너리 형식
//   model.load_weights("my_mlp.bin")  — 바이너리에서 파라미터 복원
//
// 바이너리 포맷 (little-endian):
//   파라미터 수 (u32)
//   각 파라미터: ndim(u32) + shape(u32 × ndim) + data(f64 × 원소 수)
//
// 핵심 원리:
//   Model::params()가 결정적 순서로 파라미터를 반환:
//     layers[0].W → layers[0].b → layers[1].W → layers[1].b → ...
//   저장과 로드가 같은 순서이므로 1:1 대응이 보장됨.
//
//   Variable이 Rc<RefCell<>>로 공유되므로:
//     params()[i].set_data(loaded) → 레이어 원본 파라미터도 자동 갱신
//
// 파일 크기 (MLP(784→1000→10)):
//   W1: 784×1000 = 784,000개 f64 = 6,272,000 바이트
//   b1: 1000개 f64 = 8,000 바이트
//   W2: 1000×10 = 10,000개 f64 = 80,000 바이트
//   b2: 10개 f64 = 80 바이트
//   + 헤더 ≈ 6.1 MB

use dezero::{
    softmax_cross_entropy_simple, DataLoader, Dataset, Model, MNIST, MLP, SGD,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_weights() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 3; // step51(5에폭)보다 적음: 저장 후 이어서 학습하는 시나리오
        let batch_size = 100;
        let hidden_size = 1000;
        let lr = 0.01;

        let weight_path = "/tmp/dezero_test_my_mlp.bin"; // 테스트용 임시 경로

        // --- MNIST 로드 ---
        println!("Loading MNIST train set...");
        let train_set = MNIST::new(true);
        let mut train_loader = DataLoader::new(&train_set, batch_size, true);

        // --- 모델 & 옵티마이저 ---
        let model = MLP::new(&[hidden_size, 10]);
        let optimizer = SGD::new(lr).setup(&model);

        // --- 저장된 가중치가 있으면 로드 ---
        // Python: if os.path.exists('my_mlp.npz'): model.load_weights('my_mlp.npz')
        // 첫 실행: 파일 없음 → 스킵 (랜덤 초기화로 학습)
        // 이후 실행: 파일 있음 → 로드 후 이어서 학습
        //
        // 주의: load_weights 전에 forward를 한 번 호출해야 함
        //       → lazy init으로 W가 생성되어야 params()에 포함됨
        //       → 그래야 저장된 파라미터 수와 일치

        // dummy forward로 lazy init 트리거
        // 첫 배치 하나를 꺼내서 forward → W1(784, 1000), W2(1000, 10) 생성
        let (first_x, _) = train_loader.next().unwrap();
        let _ = model.forward(&first_x);
        train_loader.reset(); // 다시 처음부터

        if std::path::Path::new(weight_path).exists() {
            println!("Loading weights from {}", weight_path);
            model.load_weights(weight_path);
        }

        // --- 학습 루프 ---
        for epoch in 0..max_epoch {
            let mut sum_loss = 0.0;

            for (x, t) in &mut train_loader {
                let y = model.forward(&x);
                let loss = softmax_cross_entropy_simple(&y, &t);

                model.cleargrads();
                loss.backward(false, false);
                optimizer.update();

                let loss_val = loss.data().iter().next().copied().unwrap();
                sum_loss += loss_val * t.len() as f64;
            }

            println!(
                "epoch: {}, loss: {:.4}",
                epoch + 1,
                sum_loss / train_set.len() as f64
            );

            train_loader.reset();
        }

        // --- 가중치 저장 ---
        model.save_weights(weight_path);

        // 파일 크기 확인
        let file_size = std::fs::metadata(weight_path).unwrap().len();
        println!(
            "Saved weights to {} ({:.2} MB)",
            weight_path,
            file_size as f64 / 1_048_576.0
        );

        // --- 검증: 저장 후 로드하여 같은 파라미터인지 확인 ---
        let model2 = MLP::new(&[hidden_size, 10]);
        // lazy init 트리거
        let (first_x, _) = DataLoader::new(&train_set, batch_size, false).next().unwrap();
        let _ = model2.forward(&first_x);

        model2.load_weights(weight_path);

        // 원본과 로드된 모델의 파라미터가 동일한지 검증
        let params1 = model.params();
        let params2 = model2.params();
        for (i, (p1, p2)) in params1.iter().zip(params2.iter()).enumerate() {
            let d1 = p1.data();
            let d2 = p2.data();
            assert_eq!(d1.shape(), d2.shape(), "param {} shape mismatch", i);
            let max_diff: f64 = d1
                .iter()
                .zip(d2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff < 1e-15,
                "param {} data mismatch: max_diff = {}",
                i,
                max_diff
            );
        }
        println!("Verification passed: saved and loaded weights are identical");

        // 임시 파일 정리
        let _ = std::fs::remove_file(weight_path);
    }
}
