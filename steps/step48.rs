// step48: 미니배치를 사용한 스파이럴 데이터셋 학습
//
// step47과의 차이:
//   step47: softmax, cross-entropy 함수를 구현하고 단일 순전파/역전파만 검증
//   step48: 실제 데이터셋(spiral)으로 미니배치 학습을 수행하여 분류 모델 훈련
//
// 새로운 개념:
//   1) Spiral 데이터셋: 3클래스 나선형 데이터 (300개 샘플, 2차원 입력)
//      원점에서 바깥으로 뻗어나가는 3개의 나선 팔(arm)
//      단순 직선으로는 분리 불가 → 비선형 모델(MLP)이 필요
//
//   2) 미니배치(mini-batch) 학습:
//      전체 300개를 한 번에 처리하면 → 1번 업데이트 per epoch
//      30개씩 10번 처리하면 → 10번 업데이트 per epoch (더 빠르게 수렴)
//      실제 데이터가 수백만 개일 때는 전체를 한 번에 처리하는 것 자체가 불가능
//
//   3) 에폭(epoch): 전체 데이터를 한 바퀴 도는 단위
//      300개 데이터, 배치 30이면 → 1에폭 = 10 이터레이션
//      300에폭 = 3000 이터레이션
//
//   4) 데이터 셔플: 매 에폭마다 인덱스를 무작위로 섞음
//      항상 같은 배치 구성이면 학습이 특정 패턴에 편향될 수 있음
//      셔플로 매번 다른 조합을 보여주면 일반화 성능 향상

use dezero::{get_spiral, softmax_cross_entropy_simple, Model, Variable, MLP, SGD};

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

    /// Fisher-Yates 셔플: 인덱스 배열을 무작위로 섞음
    /// Python의 np.random.permutation에 해당
    fn shuffle(indices: &mut [usize], rng: &mut SimpleRng) {
        let n = indices.len();
        for i in (1..n).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
    }

    /// 2D 배열에서 지정된 행(row)들을 추출
    /// Python의 x[batch_index]에 해당 (fancy indexing)
    fn batch_select(x: &ndarray::ArrayD<f64>, indices: &[usize]) -> ndarray::ArrayD<f64> {
        let cols = x.shape()[1];
        let mut batch = ndarray::ArrayD::zeros(ndarray::IxDyn(&[indices.len(), cols]));
        for (bi, &idx) in indices.iter().enumerate() {
            for j in 0..cols {
                batch[[bi, j]] = x[[idx, j]];
            }
        }
        batch
    }

    #[test]
    fn test_spiral_training() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 300;
        let batch_size = 30;
        let hidden_size = 10;
        let lr = 1.0;

        // --- Spiral 데이터셋 ---
        // 3클래스 × 100개 = 300개 샘플, 각 샘플은 2D 좌표
        let (x_data, t_data) = get_spiral(true);
        let data_size = x_data.shape()[0]; // 300
        // ceil(300 / 30) = 10 이터레이션 per epoch
        let max_iter = (data_size + batch_size - 1) / batch_size;

        // --- 모델 & 옵티마이저 ---
        // 입력 2 → 은닉 10 (sigmoid) → 출력 3 (클래스 수)
        let model = MLP::new(&[hidden_size, 3]);
        let optimizer = SGD::new(lr).setup(&model);

        let mut rng = SimpleRng::new(0);

        for epoch in 0..max_epoch {
            // 매 에폭 시작: 인덱스 셔플
            // [0, 1, 2, ..., 299] → [142, 7, 253, ...]
            let mut index: Vec<usize> = (0..data_size).collect();
            shuffle(&mut index, &mut rng);

            let mut sum_loss = 0.0;

            for i in 0..max_iter {
                // 미니배치 추출
                let start = i * batch_size;
                let end = ((i + 1) * batch_size).min(data_size);
                let batch_index = &index[start..end];

                let batch_x = Variable::new(batch_select(&x_data, batch_index));
                let batch_t: Vec<usize> = batch_index.iter().map(|&idx| t_data[idx]).collect();

                // 순전파 → 손실 → 역전파 → 업데이트
                let y = model.forward(&batch_x);
                let loss = softmax_cross_entropy_simple(&y, &batch_t);

                model.cleargrads();
                loss.backward(false, false);
                optimizer.update();

                let loss_val = loss.data().iter().next().copied().unwrap();
                sum_loss += loss_val * batch_t.len() as f64;
            }

            // 에폭별 평균 손실 출력
            let avg_loss = sum_loss / data_size as f64;
            if epoch % 30 == 0 || epoch == max_epoch - 1 {
                println!("epoch {}, loss {:.4}", epoch + 1, avg_loss);
            }
        }

        // --- 최종 검증 ---
        // 전체 데이터에 대한 손실로 수렴 확인
        let y = model.forward(&Variable::new(x_data.clone()));
        let final_loss = softmax_cross_entropy_simple(&y, &t_data);
        let final_loss_val = final_loss.data().iter().next().copied().unwrap();
        println!("Final loss: {:.4}", final_loss_val);

        assert!(
            final_loss_val < 0.5,
            "loss should converge below 0.5, got {}",
            final_loss_val
        );
    }
}
