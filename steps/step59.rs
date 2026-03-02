// step59: RNN과 Truncated BPTT로 사인 곡선 예측
//
// step57까지는 CNN (공간적 패턴)을 다뤘고, step59부터는 RNN (시간적 패턴)을 다룬다.
//
// RNN (Recurrent Neural Network):
//   은닉 상태 h가 시간 스텝 간에 전달되어 "기억"을 유지
//   h_new = tanh(x @ W_x + b + h @ W_h)
//
//   시간 전개 (unfolding):
//     x₀ → [RNN] → h₁ → [Linear] → y₁
//            ↓
//     x₁ → [RNN] → h₂ → [Linear] → y₂
//            ↓
//     x₂ → [RNN] → h₃ → [Linear] → y₃
//
// BPTT (Backpropagation Through Time):
//   RNN을 시간 축으로 펼치면 일반 순전파 네트워크와 동일
//   → 일반적인 역전파를 그대로 적용할 수 있음
//
// Truncated BPTT:
//   시퀀스가 길면 BPTT의 계산 비용이 무한히 증가
//   해결: 일정 스텝(bptt_length)마다 역전파를 끊음
//     1. bptt_length 스텝 동안 forward + loss 누적
//     2. loss.backward() → 기울기 계산
//     3. loss.unchain_backward() → 그래프 절단
//     4. optimizer.update() → 파라미터 갱신
//   은닉 상태 h의 "값"은 유지되지만, 그래프 연결이 끊김
//   → 다음 세그먼트에서 h는 상수로 취급됨
//
// Adam 옵티마이저:
//   SGD보다 수렴이 빠름. 각 파라미터별 적응적 학습률 사용.
//   m = β₁·m + (1-β₁)·grad        (1차 모멘트)
//   v = β₂·v + (1-β₂)·grad²       (2차 모멘트)
//   p ← p - lr_t · m / (√v + ε)

use dezero::{mean_squared_error, no_grad, Adam, Linear, Variable, RNN, SinCurve};

/// SimpleRNN: RNN + Linear으로 시계열 예측
struct SimpleRNN {
    rnn: RNN,
    fc: Linear,
}

impl SimpleRNN {
    fn new(hidden_size: usize, out_size: usize) -> Self {
        SimpleRNN {
            rnn: RNN::new(hidden_size),
            fc: Linear::new(out_size, 55),
        }
    }

    fn reset_state(&self) {
        self.rnn.reset_state();
    }

    fn forward(&self, x: &Variable) -> Variable {
        let h = self.rnn.forward(x);
        self.fc.forward(&h)
    }

    fn cleargrads(&self) {
        self.rnn.cleargrads();
        self.fc.cleargrads();
    }

    fn params(&self) -> Vec<Variable> {
        let mut params = self.rnn.params();
        params.extend(self.fc.params());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_sincurve() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 20;
        let hidden_size = 100;
        let bptt_length = 30;

        // --- 데이터 ---
        let train_set = SinCurve::new(true);
        let seqlen = train_set.len();
        println!("SinCurve train set: {} samples", seqlen);

        // --- 모델 & 옵티마이저 ---
        let model = SimpleRNN::new(hidden_size, 1);
        let optimizer = Adam::new(0.001);

        // --- 학습 루프 ---
        let mut last_loss = 0.0;

        for epoch in 0..max_epoch {
            model.reset_state();
            let mut loss = Variable::new(ndarray::arr0(0.0).into_dyn());
            let mut count = 0;

            for i in 0..seqlen {
                let (x_val, t_val) = train_set.get(i);
                // (1, 1) shape: 배치=1, 입력차원=1
                let x = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![x_val]).unwrap(),
                );
                let t = Variable::new(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![t_val]).unwrap(),
                );

                let y = model.forward(&x);
                loss = &loss + &mean_squared_error(&y, &t);
                count += 1;

                // Truncated BPTT: bptt_length 스텝마다 역전파 + 그래프 절단
                if count % bptt_length == 0 || count == seqlen {
                    model.cleargrads();
                    loss.backward(false, false);
                    loss.unchain_backward();
                    optimizer.update(&model.params());
                }
            }

            let avg_loss = loss.data().iter().next().copied().unwrap() / count as f64;

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                println!("epoch {:3} | loss {:.6}", epoch + 1, avg_loss);
            }

            last_loss = avg_loss;
        }

        // loss가 감소했는지 확인
        println!("final loss: {:.6}", last_loss);
        assert!(
            last_loss < 0.1,
            "loss should decrease significantly, got {}",
            last_loss
        );

        // --- 추론 모드 ---
        // 학습된 RNN으로 사인 곡선 예측
        model.reset_state();
        let _guard = no_grad();

        let num_points = 50;
        let mut predictions = Vec::new();
        for i in 0..num_points {
            let x_val = (2.0 * std::f64::consts::PI * i as f64 / 25.0).cos();
            let x = Variable::new(
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![x_val]).unwrap(),
            );
            let y = model.forward(&x);
            predictions.push(y.data().iter().next().copied().unwrap());
        }

        println!(
            "predictions (first 10): {:?}",
            &predictions[..10]
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );

        println!("All RNN tests passed!");
    }
}
