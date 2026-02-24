// step51: MNIST 손글씨 숫자 분류
//
// step50과의 차이:
//   step50: Spiral 토이 데이터셋 (300개, 2D, 3클래스)
//   step51: MNIST 실제 데이터셋 (60000/10000개, 784D, 10클래스)
//
// MNIST (Modified National Institute of Standards and Technology):
//   - 손으로 쓴 숫자 0~9의 28×28 그레이스케일 이미지
//   - 머신러닝의 "Hello World" — 가장 유명한 벤치마크 데이터셋
//   - train: 60000개, test: 10000개
//   - 각 이미지를 784차원 벡터로 펼침 (28×28 → 784)
//   - 픽셀값을 [0, 255] → [0, 1]로 정규화 (/255.0)
//
// 학습 설정:
//   - 모델: MLP(784 → 1000 → 10), sigmoid 활성화
//   - 옵티마이저: SGD(lr=0.01)
//   - 5 에폭 (Spiral의 300 에폭과 달리 데이터가 200배 많으므로 적은 에폭으로도 학습)
//   - 배치 크기: 100
//
// step48~50에서 만든 인프라가 그대로 재사용됨:
//   Dataset 트레잇 → MNIST가 구현
//   DataLoader → 배치 순회
//   MLP → lazy init으로 784 입력 자동 처리
//   SGD, softmax_cross_entropy_simple, accuracy, no_grad → 그대로

use dezero::{
    accuracy, no_grad, softmax_cross_entropy_simple, DataLoader, Dataset, Model, MNIST, MLP, SGD,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_training() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 5; // Spiral은 300에폭이었지만, MNIST는 데이터가 200배 많으므로 5에폭이면 충분
        let batch_size = 100; // 60000 / 100 = 600 이터레이션/에폭
        let hidden_size = 1000; // 784D 입력을 처리하기 위해 큰 은닉층
        let lr = 0.01; // Spiral(lr=1.0)보다 낮음: 784D 입력 → gradient가 크므로 작은 lr 필요

        // --- MNIST 데이터셋 로드 ---
        // 첫 실행: 인터넷에서 .gz 파일 다운로드 → ~/.dezero/mnist/에 저장
        // 이후 실행: 캐시에서 즉시 로드 (다운로드 스킵)
        println!("Loading MNIST train set...");
        let train_set = MNIST::new(true); // 60000개, 각 784차원
        println!("Loading MNIST test set...");
        let test_set = MNIST::new(false); // 10000개, 각 784차원
        println!(
            "train: {} samples, test: {} samples",
            train_set.len(),
            test_set.len()
        );

        // DataLoader: step50과 동일한 인터페이스
        // Spiral → MNIST로 바꿔도 DataLoader 코드는 변경 불필요
        let mut train_loader = DataLoader::new(&train_set, batch_size, true); // shuffle=true
        let mut test_loader = DataLoader::new(&test_set, batch_size, false); // shuffle=false

        // --- 모델 & 옵티마이저 ---
        // MLP::new(&[1000, 10]): 은닉 1000, 출력 10(숫자 0~9)
        // 입력 크기 784는 명시하지 않아도 됨 → lazy init이 첫 forward에서 자동 결정
        //   첫 forward: x.shape = (100, 784) → W1: (784, 1000) 생성 (Xavier 초기화)
        // 총 파라미터: 784×1000 + 1000 + 1000×10 + 10 = 795,010개
        let model = MLP::new(&[hidden_size, 10]);
        let optimizer = SGD::new(lr).setup(&model);

        // --- 학습 루프 ---
        // step50과 완전히 동일한 구조.
        // Spiral → MNIST로 데이터셋만 바꿨을 뿐, 학습 코드는 한 글자도 다르지 않음.
        // 이것이 step49~50에서 만든 추상화(Dataset, DataLoader)의 효과.
        for epoch in 0..max_epoch {
            // --- 학습 ---
            let mut sum_loss = 0.0;
            let mut sum_acc = 0.0;

            // 600 이터레이션 (60000 / 100)
            for (x, t) in &mut train_loader {
                let y = model.forward(&x); // (100, 784) → (100, 10)
                let loss = softmax_cross_entropy_simple(&y, &t);
                let acc = accuracy(&y, &t);

                model.cleargrads();
                loss.backward(false, false);
                optimizer.update(); // SGD: p ← p - 0.01 × grad

                let loss_val = loss.data().iter().next().copied().unwrap();
                sum_loss += loss_val * t.len() as f64;
                sum_acc += acc * t.len() as f64;
            }

            let train_size = train_set.len() as f64;
            println!("epoch: {}", epoch + 1);
            println!(
                "train loss: {:.4}, accuracy: {:.4}",
                sum_loss / train_size,
                sum_acc / train_size
            );

            // --- 테스트 (no_grad) ---
            // 평가만 하므로 역전파 그래프 불필요 → 메모리 절약 + 속도 향상
            let mut test_loss = 0.0;
            let mut test_acc = 0.0;

            {
                let _guard = no_grad(); // RAII: 스코프 끝에서 자동 복원
                for (x, t) in &mut test_loader {
                    let y = model.forward(&x);
                    let loss = softmax_cross_entropy_simple(&y, &t);
                    let acc = accuracy(&y, &t);
                    test_loss += loss.data().iter().next().copied().unwrap() * t.len() as f64;
                    test_acc += acc * t.len() as f64;
                }
            }

            let test_size = test_set.len() as f64;
            println!(
                "test loss: {:.4}, accuracy: {:.4}",
                test_loss / test_size,
                test_acc / test_size
            );

            train_loader.reset(); // 다음 에폭 준비 (인덱스 재셔플)
            test_loader.reset();
        }
    }
}
