// step45: Model로 여러 레이어를 하나로 묶기
// step44와의 차이:
//   step44: l1, l2를 개별 변수로 가지고 각각 cleargrads(), params() 호출
//           → l1.cleargrads(); l2.cleargrads();
//           → for l in [&l1, &l2] { for p in l.params() { ... } }
//   step45: TwoLayerNet이 l1, l2를 내부에 묶고, Model 트레잇이 일괄 처리
//           → model.cleargrads();  (한 번으로 모든 레이어의 기울기 초기화)
//           → for p in model.params() { ... }  (모든 파라미터를 일괄 순회)
//
// 왜 Model이 필요한가?
//   step44에서 레이어가 2개일 때: l1.cleargrads(); l2.cleargrads();
//   레이어가 10개면? l1~l10까지 일일이 호출해야 함.
//   Model로 묶으면 layers()에 한 번 등록해두고
//   cleargrads(), params()를 모델 단위로 호출할 수 있다.
//
// 추상화의 단계:
//   step43: 파라미터(W, b) 개별 관리        → 변수 4개
//   step44: Layer로 파라미터 묶기            → 레이어 2개
//   step45: Model로 레이어 묶기             → 모델 1개

use dezero::{mean_squared_error, sigmoid, Linear, Model, Variable};

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

    // 사용자 정의 모델: 2층 신경망
    // Python에서는 class TwoLayerNet(Model):로 상속
    // Rust에서는 struct + impl Model 트레잇으로 구현
    struct TwoLayerNet {
        l1: Linear,
        l2: Linear,
    }

    impl TwoLayerNet {
        fn new(hidden_size: usize, out_size: usize) -> Self {
            TwoLayerNet {
                l1: Linear::new(hidden_size, 42),
                l2: Linear::new(out_size, 43),
            }
        }
    }

    impl Model for TwoLayerNet {
        fn forward(&self, x: &Variable) -> Variable {
            let y = sigmoid(&self.l1.forward(x));
            self.l2.forward(&y)
        }

        // 모델이 가진 모든 레이어를 나열
        // Model 트레잇의 cleargrads()와 params()가 이 목록을 사용
        fn layers(&self) -> Vec<&Linear> {
            vec![&self.l1, &self.l2]
        }
    }

    #[test]
    fn test_neural_network_with_model() {
        let mut rng = SimpleRng::new(0);

        // 토이 데이터셋: y = sin(2πx) + noise (step43, 44와 동일)
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin() + rng.next_f64())
            .collect();

        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, 1]), x_data).unwrap(),
        );
        let y = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, 1]), y_data).unwrap(),
        );

        // step44: let l1 = Linear::new(10, 42); let l2 = Linear::new(1, 43);
        // step45: 모델 하나로 묶음
        let model = TwoLayerNet::new(10, 1);

        let lr = 0.2;
        let iters = 10000;

        for i in 0..iters {
            // 순전파: model.forward()가 l1 → sigmoid → l2 전체를 처리
            let y_pred = model.forward(&x);
            let loss = mean_squared_error(&y, &y_pred);

            // step44: l1.cleargrads(); l2.cleargrads();
            // step45: 모델 단위로 한 번에 처리 (Model 트레잇의 기본 구현)
            model.cleargrads();
            loss.backward(false, false);

            // step44: for l in [&l1, &l2] { for p in l.params() { ... } }
            // step45: model.params()가 모든 레이어의 파라미터를 일괄 반환
            for p in model.params() {
                let grad = p.grad().unwrap();
                p.set_data(&p.data() - &grad.mapv(|v| v * lr));
            }

            if i % 1000 == 0 {
                println!("iter {}: loss = {}", i, loss);
            }
        }

        // 최종 손실 확인
        let y_pred = model.forward(&x);
        let final_loss = mean_squared_error(&y, &y_pred);
        let loss_val = final_loss.data().iter().next().copied().unwrap();
        println!("Final loss: {:.6}", loss_val);

        assert!(loss_val < 0.2, "loss should converge, got {}", loss_val);
    }
}
