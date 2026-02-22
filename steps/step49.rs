// step49: Dataset 클래스로 데이터셋 추상화
//
// step48과의 차이:
//   step48: get_spiral() 함수가 (ArrayD, Vec<usize>) 튜플을 직접 반환
//           → 배치 추출 시 batch_select() 같은 헬퍼를 직접 만들어야 함
//           → 데이터셋마다 반환 형태가 달라질 수 있음
//   step49: Spiral이 Dataset 트레잇을 구현
//           → train_set.len(), train_set.get(i) 로 통일된 인터페이스
//           → 어떤 데이터셋이든 같은 학습 루프 코드를 재사용 가능
//
// 왜 Dataset 추상화가 필요한가?
//   현재는 Spiral 하나뿐이지만, 실제로는 MNIST, CIFAR-10 등 다양한 데이터셋을 사용.
//   Dataset 트레잇으로 통일하면:
//     let train_set = Spiral::new(true);   → let train_set = MNIST::new(true);
//   이 한 줄만 바꿔도 나머지 학습 코드는 그대로 동작.
//
// 추상화의 단계:
//   step48: 함수 → 원시 데이터 → 수동 배치 추출
//   step49: Dataset 트레잇 → 통일된 인터페이스 → get()으로 개별 샘플 접근

use dezero::{
    softmax_cross_entropy_simple, Dataset, Model, Spiral, Variable, MLP, SGD,
};

#[cfg(test)]
mod tests {
    use super::*;

    // 간단한 시드 기반 난수 생성기 (LCG)
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            SimpleRng { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Fisher-Yates 셔플
    fn shuffle(indices: &mut [usize], rng: &mut SimpleRng) {
        let n = indices.len();
        for i in (1..n).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
    }

    #[test]
    fn test_spiral_with_dataset() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 300;
        let batch_size = 30;
        let hidden_size = 10;
        let lr = 1.0;

        // step48: let (x_data, t_data) = get_spiral(true);
        // step49: Dataset 트레잇을 구현하는 Spiral 구조체
        //         → len()과 get()으로 통일된 인터페이스 제공
        let train_set = Spiral::new(true);

        // step48: let data_size = x_data.shape()[0];
        // step49: Dataset 트레잇의 len()
        let data_size = train_set.len();
        let max_iter = (data_size + batch_size - 1) / batch_size;

        let model = MLP::new(&[hidden_size, 3]);
        let optimizer = SGD::new(lr).setup(&model);

        let mut rng = SimpleRng::new(0);

        for epoch in 0..max_epoch {
            let mut index: Vec<usize> = (0..data_size).collect();
            shuffle(&mut index, &mut rng);

            let mut sum_loss = 0.0;

            for i in 0..max_iter {
                let start = i * batch_size;
                let end = ((i + 1) * batch_size).min(data_size);
                let batch_index = &index[start..end];

                // step48: batch_select(&x_data, batch_index) + 수동 라벨 추출
                // step49: Dataset.get()으로 개별 샘플 접근 후 배치 조립
                //   Python: batch = [train_set[i] for i in batch_index]
                //           batch_x = np.array([example[0] for example in batch])
                //           batch_t = np.array([example[1] for example in batch])
                let mut batch_x_data = Vec::new();
                let mut batch_t = Vec::new();
                for &idx in batch_index {
                    let (x, t) = train_set.get(idx);
                    batch_x_data.extend_from_slice(&x);
                    batch_t.push(t);
                }
                let batch_x = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[batch_index.len(), 2]),
                        batch_x_data,
                    )
                    .unwrap(),
                );

                let y = model.forward(&batch_x);
                let loss = softmax_cross_entropy_simple(&y, &batch_t);

                model.cleargrads();
                loss.backward(false, false);
                optimizer.update();

                let loss_val = loss.data().iter().next().copied().unwrap();
                sum_loss += loss_val * batch_t.len() as f64;
            }

            let avg_loss = sum_loss / data_size as f64;
            if epoch % 30 == 0 || epoch == max_epoch - 1 {
                println!("epoch {}, loss {:.4}", epoch + 1, avg_loss);
            }
        }

        // --- 최종 검증 ---
        // 전체 데이터에 대한 손실 확인
        let mut all_x = Vec::new();
        let mut all_t = Vec::new();
        for i in 0..data_size {
            let (x, t) = train_set.get(i);
            all_x.extend_from_slice(&x);
            all_t.push(t);
        }
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data_size, 2]), all_x).unwrap(),
        );
        let loss = softmax_cross_entropy_simple(&model.forward(&x), &all_t);
        let final_loss = loss.data().iter().next().copied().unwrap();
        println!("Final loss: {:.4}", final_loss);

        assert!(
            final_loss < 0.5,
            "loss should converge below 0.5, got {}",
            final_loss
        );
    }
}
