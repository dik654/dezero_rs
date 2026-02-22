// step44: 레이어(Layer)로 파라미터 관리
// step43과의 차이:
//   step43: W1, b1, W2, b2를 개별 Variable로 선언하고 관리
//           → cleargrad 4번, set_data 4번, 변수 선언 4개
//   step44: Linear 레이어가 W, b를 내부에 묶어서 관리
//           → l1.cleargrads(), l2.cleargrads()로 간결해짐
//           → for p in l.params()로 일괄 업데이트
//
// 왜 Layer가 필요한가?
//   2층 네트워크에서도 파라미터가 4개(W1,b1,W2,b2)인데
//   10층이면 20개. 개별 관리는 실수하기 쉽고 코드가 길어짐.
//   Layer로 묶으면 "레이어 단위"로 관리할 수 있다.
//
// Lazy initialization (지연 초기화):
//   Linear::new(10, seed)에서 출력 크기(10)만 지정.
//   W의 입력 크기는 첫 forward 호출 시 x.shape[1]에서 자동 결정.
//   → 사용자가 입출력 크기를 일일이 맞출 필요 없음

use dezero::{mean_squared_error, sigmoid, Linear, Variable};

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
    fn test_neural_network_with_layer() {
        let mut rng = SimpleRng::new(0);

        // 토이 데이터셋: y = sin(2πx) + noise (step43과 동일)
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

        // step43에서는 W1, b1, W2, b2를 직접 생성하고 초기화했음:
        //   let w1 = Variable::new(shape [1, 10], 0.01 * randn);
        //   let b1 = Variable::new(shape [10], zeros);
        //   let w2 = Variable::new(shape [10, 1], 0.01 * randn);
        //   let b2 = Variable::new(shape [1], zeros);
        //
        // step44: 출력 크기만 지정하면 Linear 레이어가 나머지를 처리
        //   W의 입력 크기는 첫 forward에서 자동 결정 (lazy initialization)
        //   W는 Xavier 초기화: randn * sqrt(1/in_size)
        let l1 = Linear::new(10, 42); // 입력 ? → 출력 10
        let l2 = Linear::new(1, 43); // 입력 10 → 출력 1

        let lr = 0.2;
        let iters = 10000;

        for i in 0..iters {
            // 순전파: step43과 동일한 구조이지만 레이어 호출로 간결
            // step43: let h = linear(&x, &w1, Some(&b1));
            // step44: let h = l1.forward(&x);
            let h = l1.forward(&x);
            let h = sigmoid(&h);
            let y_pred = l2.forward(&h);

            let loss = mean_squared_error(&y, &y_pred);

            // step43: w1.cleargrad(); b1.cleargrad(); w2.cleargrad(); b2.cleargrad();
            // step44: 레이어 단위로 한 번에 처리
            l1.cleargrads();
            l2.cleargrads();
            loss.backward(false, false);

            // step43: w1.set_data(...); b1.set_data(...); w2.set_data(...); b2.set_data(...);
            // step44: 레이어의 params()로 순회하며 일괄 업데이트
            //   params()가 반환하는 Variable은 레이어 내부와 같은 Rc를 공유하므로
            //   여기서 set_data하면 레이어 내부의 W, b도 함께 바뀜
            for l in [&l1, &l2] {
                for p in l.params() {
                    let grad = p.grad().unwrap();
                    p.set_data(&p.data() - &grad.mapv(|v| v * lr));
                }
            }

            if i % 1000 == 0 {
                println!("iter {}: loss = {}", i, loss);
            }
        }

        // 최종 손실 확인
        let h = l1.forward(&x);
        let h = sigmoid(&h);
        let y_pred = l2.forward(&h);
        let final_loss = mean_squared_error(&y, &y_pred);
        let loss_val = final_loss.data().iter().next().copied().unwrap();
        println!("Final loss: {:.6}", loss_val);

        assert!(loss_val < 0.2, "loss should converge, got {}", loss_val);
    }
}
