// step60: LSTM으로 사인 곡선 예측 (배치 학습)
//
// step59의 SimpleRNN 한계:
//   - RNN은 시퀀스가 길어지면 기울기 소실/폭발 (vanishing/exploding gradient)
//   - h_new = tanh(x@W + h@W) → tanh를 반복 적용하면 기울기가 0 또는 ∞로 발산
//   - 장기 의존성(long-term dependency) 학습 불가
//
// LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997):
//   셀 상태(cell state)를 별도로 유지하여 장기 기억을 보존
//
//   4개 게이트:
//     f = σ(x@W_xf + h@W_hf + b_f)   — forget: 과거 기억 중 잊을 비율
//     i = σ(x@W_xi + h@W_hi + b_i)   — input: 새 정보 중 기억할 비율
//     g = tanh(x@W_xg + h@W_hg + b_g) — candidate: 새로운 후보 기억
//     o = σ(x@W_xo + h@W_ho + b_o)   — output: 출력할 비율
//
//   상태 업데이트:
//     c_new = f ⊙ c + i ⊙ g   — 선택적 망각 + 선택적 기억
//     h_new = o ⊙ tanh(c_new)  — 셀 상태를 필터링한 출력
//
//   기울기 흐름: c를 통해 덧셈으로 전파 → 곱셈 반복이 아니므로 기울기 보존
//
// SeqDataLoader:
//   시계열 데이터를 batch_size개 병렬 스트림으로 분할하여 배치 학습
//   일반 DataLoader: 랜덤 셔플 (분류)
//   SeqDataLoader: 시간 순서 유지 (시계열)

use dezero::{mean_squared_error, no_grad, Adam, Linear, Variable, LSTM, SeqDataLoader, SinCurve};

/// BetterRNN: LSTM + Linear으로 시계열 예측
/// step59의 SimpleRNN을 LSTM으로 교체
struct BetterRNN {
    rnn: LSTM,
    fc: Linear,
}

impl BetterRNN {
    fn new(hidden_size: usize, out_size: usize) -> Self {
        BetterRNN {
            rnn: LSTM::new(hidden_size),
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
    fn test_lstm_sincurve() {
        // --- 하이퍼파라미터 ---
        let max_epoch = 15;
        let batch_size = 30;
        let hidden_size = 100;
        let bptt_length = 30;

        // --- 데이터 ---
        let train_set = SinCurve::new(true);
        let mut dataloader = SeqDataLoader::new(&train_set, batch_size);
        let seqlen = dataloader.jump; // 배치당 시간 스텝 수
        println!(
            "SinCurve: {} samples, batch_size={}, {} steps/epoch",
            train_set.len(),
            batch_size,
            seqlen
        );

        // --- 모델 & 옵티마이저 ---
        let model = BetterRNN::new(hidden_size, 1);
        let optimizer = Adam::new(0.001);

        // --- 학습 루프 ---
        let mut last_loss = 0.0;

        for epoch in 0..max_epoch {
            model.reset_state();
            let mut loss = Variable::new(ndarray::arr0(0.0).into_dyn());
            let mut count = 0;

            for (x, t) in &mut dataloader {
                let y = model.forward(&x);
                loss = &loss + &mean_squared_error(&y, &t);
                count += 1;

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
            dataloader.reset();
        }

        // loss가 감소했는지 확인
        println!("final loss: {:.6}", last_loss);
        assert!(
            last_loss < 1.0,
            "loss should decrease, got {}",
            last_loss
        );

        // --- 추론 ---
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
            "predictions (first 5): {:?}",
            &predictions[..5]
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );

        println!("All LSTM tests passed!");
    }
}
