// step46: MLP와 SGD 옵티마이저
//
// step45와의 차이:
//   1) TwoLayerNet(사용자 정의 모델) → MLP(범용 다층 모델)
//      step45: struct TwoLayerNet { l1, l2 }, impl Model for TwoLayerNet { ... }
//      step46: MLP::new(&[10, 1])  ← 한 줄로 동일한 구조 생성
//
//   2) 수동 파라미터 업데이트 → SGD 옵티마이저
//      step45: for p in model.params() { p.set_data(&p.data() - &grad * lr); }
//      step46: optimizer.update();  ← 옵티마이저가 업데이트 로직을 캡슐화
//
// 왜 옵티마이저가 필요한가?
//   SGD는 가장 단순한 업데이트: p ← p - lr × grad
//   하지만 실제로는 Momentum, Adam 등 다양한 업데이트 규칙이 있다.
//   업데이트 로직을 옵티마이저로 분리하면
//   optimizer = SGD(lr) → optimizer = Adam(lr) 한 줄만 바꿔서 전환 가능.
//
// 추상화의 단계:
//   step43: 파라미터 개별 관리 (W1, b1, W2, b2)
//   step44: Layer로 파라미터 묶기 (l1, l2)
//   step45: Model로 레이어 묶기 (model)
//   step46: Optimizer로 업데이트 분리 + MLP로 모델 정의 간소화

use dezero::{mean_squared_error, Model, Variable, MLP, SGD};

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

    #[test]
    fn test_mlp_with_sgd() {
        let mut rng = SimpleRng::new(0);

        // 토이 데이터셋: y = sin(2πx) + noise
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

        let lr = 0.2;
        let iters = 10000;
        let hidden_size = 10;

        // step45: let model = TwoLayerNet::new(10, 1);  (사용자 정의 구조체 필요)
        // step46: MLP::new(&[10, 1])  (범용, 한 줄로 생성)
        //   내부적으로 Linear(10) → sigmoid → Linear(1) 구조를 자동 생성
        let model = MLP::new(&[hidden_size, 1]);

        // step45: for p in model.params() { p.set_data(...); }  (수동 업데이트)
        // step46: optimizer.update()  (옵티마이저가 처리)
        let optimizer = SGD::new(lr).setup(&model);

        for i in 0..iters {
            let y_pred = model.forward(&x);
            let loss = mean_squared_error(&y, &y_pred);

            model.cleargrads();
            loss.backward(false, false);

            // step45: for p in model.params() {
            //     let grad = p.grad().unwrap();
            //     p.set_data(&p.data() - &grad.mapv(|v| v * lr));
            // }
            // step46: 한 줄로 끝
            optimizer.update();

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
