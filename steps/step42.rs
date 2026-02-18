// step42: 선형 회귀 (linear regression)
// y_pred = x @ W + b
// loss = mean_squared_error(y, y_pred)
// 경사하강법으로 W, b를 학습

use dezero::{matmul, sum, Variable};

// 예측 함수: y_pred = x @ W + b
// x (100,1) @ W (1,1) = (100,1), + b (1,) → 브로드캐스트 → (100,1)
fn predict(x: &Variable, w: &Variable, b: &Variable) -> Variable {
    &matmul(x, w) + b
}

// 평균제곱오차: sum((y - y_pred)²) / n
// 첫 호출 시 y_pred=0이므로 loss = sum(y²)/100 ≈ 41.68 (y가 5~7 범위)
fn mean_squared_error(x0: &Variable, x1: &Variable) -> Variable {
    let diff = x0 - x1;          // 각 샘플별 오차 (100,1)
    let n = diff.len();           // 첫 번째 축의 크기 = 100
    &sum(&diff.pow(2.0)) / (n as f64) // 오차 제곱의 평균 → 스칼라
}

#[cfg(test)]
mod tests {
    use super::*;

    // 간단한 시드 기반 난수 생성기 (LCG)
    // rand 의존성 없이 재현 가능한 난수 생성
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            SimpleRng { state: seed }
        }

        /// [0, 1) 범위의 f64 난수 생성
        fn next_f64(&mut self) -> f64 {
            // LCG 공식: state = (state * a + c) mod 2^64
            // wrapping_mul/add가 자연스럽게 u64 오버플로우로 mod 2^64 처리
            // a, c 값은 PCG 논문(O'Neill)에서 가져온 상수
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)  // a: 승수(multiplier)
                .wrapping_add(1442695040888963407); // c: 증분(increment)
            // 상위 53비트를 추출하여 [0, 1) 범위의 f64로 변환
            // >> 11: 하위 11비트 버림 (LCG 하위 비트는 품질이 낮음)
            // / 2^53: f64 가수부(mantissa)가 53비트이므로 정밀도 손실 없이 [0, 1)로 정규화
            (self.state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_linear_regression() {
        let mut rng = SimpleRng::new(0);

        // 토이 데이터셋 생성: y = 5 + 2*x + noise
        // noise는 [0, 1) 균등분포 → 평균 0.5
        // 따라서 모델이 학습하게 되는 실질적 관계: y ≈ 5.5 + 2*x
        //   W의 목표값 = 2.0 (x의 계수)
        //   b의 목표값 ≈ 5.5 (절편 5.0 + noise 평균 0.5)
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&xi| 5.0 + 2.0 * xi + rng.next_f64())
            .collect();

        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, 1]), x_data).unwrap(),
        );
        let y = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, 1]), y_data).unwrap(),
        );

        // 학습 파라미터 초기화 (전부 0에서 시작)
        // 첫 예측: y_pred = x @ [[0]] + [0] = 0 → loss가 큼 (41.68)
        let w = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[1, 1]))); // W=[[0.0]] 기울기
        let b = Variable::new(ndarray::ArrayD::zeros(ndarray::IxDyn(&[1])));    // b=[0.0] 절편

        let lr = 0.1;   // 학습률: 기울기 방향으로 한 번에 얼마나 이동할지
        let iters = 100; // 반복 횟수

        for i in 0..iters {
            // 1. 순전파: 현재 W, b로 예측
            let y_pred = predict(&x, &w, &b);
            // 2. 손실 계산: 예측과 정답의 차이
            let loss = mean_squared_error(&y, &y_pred);

            // 3. 기울기 초기화 (이전 반복의 기울기 제거)
            w.cleargrad();
            b.cleargrad();
            // 4. 역전파: loss에 대한 W, b의 기울기 계산
            loss.backward(false, false);

            // 5. 파라미터 업데이트: W -= lr * dL/dW, b -= lr * dL/db
            // .data를 직접 수정하여 계산 그래프에 기록되지 않도록 함
            // (파라미터 업데이트는 학습의 관리 작업이지 미분할 대상이 아니기 때문)
            let w_grad = w.grad().unwrap();
            let b_grad = b.grad().unwrap();
            w.set_data(&w.data() - &w_grad.mapv(|v| v * lr));
            b.set_data(&b.data() - &b_grad.mapv(|v| v * lr));

            if i % 10 == 0 {
                println!("iter {}: W = {}, b = {}, loss = {}", i, w, b, loss);
            }
        }

        // W ≈ 2.0, b ≈ 5.5 (5 + noise 평균 0.5) 로 수렴하는지 확인
        let w_val = w.data()[[0, 0]];
        let b_val = b.data()[[0]];
        println!("Final: W = {:.4}, b = {:.4}", w_val, b_val);

        assert!(
            (w_val - 2.0).abs() < 0.5,
            "W should be close to 2.0, got {}",
            w_val
        );
        assert!(
            (b_val - 5.5).abs() < 0.5,
            "b should be close to 5.5, got {}",
            b_val
        );
    }
}
