// step50: DataLoader와 정확도 평가
//
// step49와의 차이:
//   step49: 셔플, 배치 추출, 인덱싱을 학습 루프 안에서 수동으로 처리
//           → shuffle(), batch_select() 같은 헬퍼를 직접 구현
//           → 데이터셋을 바꿀 때마다 배치 추출 코드도 수정 필요
//   step50: DataLoader가 이 모든 것을 캡슐화
//           → for (x, t) in &mut loader { ... } 한 줄로 배치 순회
//           → accuracy() 함수로 분류 정확도 측정
//           → train/test 분리 평가 (no_grad로 테스트 시 역전파 비활성화)
//
// 새로운 개념:
//   1) DataLoader: Dataset + 배치 크기 + 셔플을 묶어서
//      Iterator 트레잇으로 배치를 하나씩 반환
//      매 에폭마다 reset()으로 처음부터 다시 순회
//
//   2) accuracy: argmax(예측) == 정답인 비율
//      학습 중에는 loss만으로 진행 상황을 파악하기 어려움
//      "300개 중 270개 맞음 = 90%" 같은 직관적 지표
//
//   3) Train/Test 분리:
//      학습 데이터로만 평가하면 "암기"도 좋은 성적이 됨
//      처음 보는 테스트 데이터에서도 잘 맞춰야 진짜 학습된 것
//      → test set 평가 시 no_grad()로 불필요한 계산 그래프 생성 방지

use dezero::{
    accuracy, no_grad, softmax_cross_entropy_simple, DataLoader, Dataset, Model, Spiral, Variable,
    MLP, SGD,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_training() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 300;
        let batch_size = 30;
        let hidden_size = 10;
        let lr = 1.0;

        // --- 데이터셋 ---
        // train: 학습용 (seed=1984), test: 평가용 (seed=2020)
        let train_set = Spiral::new(true);
        let test_set = Spiral::new(false);

        // step49: 수동 셔플 + batch_select
        // step50: DataLoader가 배치 순회를 캡슐화
        let mut train_loader = DataLoader::new(&train_set, batch_size, true); // shuffle=true
        let mut test_loader = DataLoader::new(&test_set, batch_size, false); // shuffle=false

        // --- 모델 & 옵티마이저 ---
        let model = MLP::new(&[hidden_size, 3]);
        let optimizer = SGD::new(lr).setup(&model);

        for epoch in 0..max_epoch {
            // --- 학습 ---
            let mut sum_loss = 0.0;
            let mut sum_acc = 0.0;

            // step49: for i in 0..max_iter { 수동 배치 추출 ... }
            // step50: for (x, t) in &mut train_loader { ... }
            for (x, t) in &mut train_loader {
                let y = model.forward(&x);
                let loss = softmax_cross_entropy_simple(&y, &t);
                let acc = accuracy(&y, &t);

                model.cleargrads();
                loss.backward(false, false);
                optimizer.update();

                let loss_val = loss.data().iter().next().copied().unwrap();
                sum_loss += loss_val * t.len() as f64;
                sum_acc += acc * t.len() as f64;
            }

            if epoch % 30 == 0 || epoch == max_epoch - 1 {
                let train_size = train_set.len() as f64;
                println!(
                    "epoch {}",
                    epoch + 1
                );
                println!(
                    "  train loss: {:.4}, accuracy: {:.4}",
                    sum_loss / train_size,
                    sum_acc / train_size
                );

                // --- 테스트 (no_grad) ---
                // 평가 시에는 역전파 그래프를 만들 필요가 없음
                // no_grad()로 불필요한 계산을 절약
                let mut test_loss = 0.0;
                let mut test_acc = 0.0;

                {
                    let _guard = no_grad();
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
                    "  test  loss: {:.4}, accuracy: {:.4}",
                    test_loss / test_size,
                    test_acc / test_size
                );

                test_loader.reset();
            }

            // 다음 에폭을 위해 리셋 (shuffle=true면 인덱스 재셔플)
            train_loader.reset();
        }

        // --- 최종 검증 ---
        // 전체 train 정확도
        let mut all_x = Vec::new();
        let mut all_t = Vec::new();
        for i in 0..train_set.len() {
            let (x, t) = train_set.get(i);
            all_x.extend_from_slice(&x);
            all_t.push(t);
        }
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[train_set.len(), 2]),
                all_x,
            )
            .unwrap(),
        );
        let _guard = no_grad();
        let y = model.forward(&x);
        let train_acc = accuracy(&y, &all_t);
        println!("\nFinal train accuracy: {:.4}", train_acc);

        assert!(
            train_acc > 0.8,
            "train accuracy should be > 80%, got {:.1}%",
            train_acc * 100.0
        );
    }
}
