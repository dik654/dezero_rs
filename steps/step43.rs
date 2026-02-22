// step43: 신경망 (neural network)
// 2층 신경망으로 비선형 회귀: y = sin(2πx) + noise
// 구조: x → linear → sigmoid → linear → y_pred
//
// step42(선형 회귀)와의 차이:
//   step42: y = x @ W + b (직선만 표현 가능)
//   step43: sigmoid 활성화 함수를 끼워 비선형(곡선) 표현 가능
//
// 왜 sigmoid가 필요한가?
//   sigmoid 없이 linear을 2번 쌓으면: y = x @ W1 @ W2 + c
//   행렬곱의 결합법칙에 의해 W1 @ W2 = W3 (하나의 행렬)로 합쳐지므로
//   결국 y = x @ W3 + c → 직선과 동일. 층을 아무리 쌓아도 직선.
//   sigmoid가 중간에 비선형 변환을 넣어야 곡선을 표현할 수 있다.

use dezero::{linear, mean_squared_error, sigmoid, Variable};

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

        /// [0, 1) 균등분포
        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.state >> 11) as f64 / (1u64 << 53) as f64
        }

        /// 표준정규분포 (Box-Muller 변환)
        /// 균등분포 u1, u2로부터 정규분포 생성:
        ///   z = sqrt(-2 * ln(u1)) * cos(2π * u2)
        fn next_normal(&mut self) -> f64 {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn test_neural_network() {
        let mut rng = SimpleRng::new(0);

        // 토이 데이터셋: y = sin(2πx) + noise
        // step42와 달리 비선형 관계 → 직선으로는 피팅 불가
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

        // 네트워크 구조: 입력 1 → 은닉 10 → 출력 1
        //
        // sigmoid 함수 형태는 항상 같은 S자이지만,
        // 1층에서 x에 곱하는 W1과 더하는 b1이 뉴런마다 다르므로
        // sigmoid에 들어가는 값이 달라져 S자의 위치와 기울기가 바뀐다:
        //   뉴런1: σ( 3x - 1) → x=0.3 근처에서 급하게 올라감
        //   뉴런2: σ(-2x + 1) → x=0.5 근처에서 내려감 (W가 음수)
        //   뉴런3: σ(0.5x + 2) → 왼쪽에서 완만하게 올라감
        //   ... (10개)
        //
        // 2층에서 이 10개 S자를 W2로 가중합:
        //   y = 0.8*뉴런1 - 1.2*뉴런2 + 0.3*뉴런3 + ... + b2
        // 서로 다른 S자를 더하고 빼면 sin 같은 곡선을 조합할 수 있다
        let (i_size, h_size, o_size) = (1, 10, 1);

        // 1층 가중치: W1 (1,10), b1 (10,)
        // 0.01을 곱해 작은 값으로 초기화 (큰 초기값은 sigmoid 포화 영역으로 가서 학습이 느려짐)
        let w1_data: Vec<f64> = (0..i_size * h_size)
            .map(|_| 0.01 * rng.next_normal())
            .collect();
        let w1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[i_size, h_size]), w1_data).unwrap(),
        );
        let b1 = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[h_size])));

        // 2층 가중치: W2 (10,1), b2 (1,)
        let w2_data: Vec<f64> = (0..h_size * o_size)
            .map(|_| 0.01 * rng.next_normal())
            .collect();
        let w2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[h_size, o_size]), w2_data).unwrap(),
        );
        let b2 = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[o_size])));

        // step42보다 lr이 크고(0.2 vs 0.1) 반복도 많음(10000 vs 100)
        // 비선형 피팅은 파라미터 공간이 복잡해서 더 많은 학습이 필요
        let lr = 0.2;
        let iters = 10000;

        for i in 0..iters {
            // 순전파: x → linear(W1,b1) → sigmoid → linear(W2,b2) → y_pred
            let h = linear(&x, &w1, Some(&b1));    // 1층: (100,1) @ (1,10) + (10,) → (100,10)
            let h = sigmoid(&h);                     // 활성화: 비선형 변환. 이게 없으면 직선밖에 안 됨
            let y_pred = linear(&h, &w2, Some(&b2)); // 2층: (100,10) @ (10,1) + (1,) → (100,1)

            // 모델이 얼마나 틀렸는지를 하나의 숫자로 표현
            // 제곱하여 양수 오차와 음수 오차가 상쇄되는 것을 방지
            // +5와 -5 → 합 = 0 → 오차 없음으로 잘못 판단
            let loss = mean_squared_error(&y, &y_pred);

            // cleargrad를 안 하면 기울기가 누적되기 때문
            w1.cleargrad();
            b1.cleargrad();
            w2.cleargrad();
            b2.cleargrad();
            // 이번 반복의 기울기만 계산
            loss.backward(false, false);

            // 파라미터 업데이트 (step42와 동일한 경사하강법, 파라미터가 4개로 늘어남)
            let w1_grad = w1.grad().unwrap();
            let b1_grad = b1.grad().unwrap();
            let w2_grad = w2.grad().unwrap();
            let b2_grad = b2.grad().unwrap();
            w1.set_data(&w1.data() - &w1_grad.mapv(|v| v * lr));
            b1.set_data(&b1.data() - &b1_grad.mapv(|v| v * lr));
            w2.set_data(&w2.data() - &w2_grad.mapv(|v| v * lr));
            b2.set_data(&b2.data() - &b2_grad.mapv(|v| v * lr));

            if i % 1000 == 0 {
                println!("iter {}: loss = {}", i, loss);
            }
        }

        // 최종 손실 확인: sin 곡선을 잘 피팅했는지
        let h = linear(&x, &w1, Some(&b1));
        let h = sigmoid(&h);
        let y_pred = linear(&h, &w2, Some(&b2));
        let final_loss = mean_squared_error(&y, &y_pred);
        let loss_val = final_loss.data().iter().next().copied().unwrap();
        println!("Final loss: {:.6}", loss_val);

        assert!(loss_val < 0.2, "loss should converge, got {}", loss_val);
    }
}
