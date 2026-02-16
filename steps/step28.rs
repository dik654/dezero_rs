// step28: Rosenbrock 함수의 경사하강법 최적화
// 반복적으로 기울기를 계산하고 변수를 갱신하여 최솟값(1, 1)에 수렴시킨다

use dezero::Variable;

fn rosenbrock(x0: &Variable, x1: &Variable) -> Variable {
    // y = 100 * (x1 - x0^2)^2 + (x0 - 1)^2
    let t1 = x1 - &x0.pow(2.0);
    let t2 = 100.0 * &t1.pow(2.0);
    let t3 = (x0 - 1.0).pow(2.0);
    &t2 + &t3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_gradient_descent() {
        let x0 = Variable::new(ndarray::arr0(0.0).into_dyn());
        let x1 = Variable::new(ndarray::arr0(2.0).into_dyn());
        // 하이퍼파라미터 학습률
        // 경사하강법 한번 실행할 때마다 기울기를 얼마나 움직일지 결정
        // 여기서는 고정된 값이지만 실전에서는 처음에는 크게 설정하여 대략적인 위치를 잡고
        // 나중에는 작게하여 최솟값 근처에서 정밀하게 조정
        let lr = 0.001;
        let iters = 1000;

        for i in 0..iters {
            let y = rosenbrock(&x0, &x1);

            x0.cleargrad();
            x1.cleargrad();
            y.backward(false, false);

            let g0 = x0.grad().unwrap();
            let g1 = x1.grad().unwrap();
            // 경사하강법 진행
            // 임의적으로 1000번 했지만 (1,1)에 수렴 못한 이유는
            // lr이 너무 작기 때문
            x0.set_data(x0.data() - lr * &g0);
            x1.set_data(x1.data() - lr * &g1);

            if i % 100 == 0 {
                println!(
                    "iter {}: x0={:.6}, x1={:.6}, y={:.6}",
                    i,
                    x0.data().iter().next().unwrap(),
                    x1.data().iter().next().unwrap(),
                    y.data().iter().next().unwrap(),
                );
            }
        }

        // Rosenbrock 최솟값은 (1, 1)에서 0
        // lr=0.001, 1000회로는 완전 수렴하지 않지만 초기값(0, 2)에서 크게 개선되어야 함
        let final_y = rosenbrock(&x0, &x1);
        let final_val = *final_y.data().iter().next().unwrap();
        println!(
            "final: x0={:.6}, x1={:.6}, y={:.6}",
            x0.data().iter().next().unwrap(),
            x1.data().iter().next().unwrap(),
            final_val,
        );

        // 초기 y=401에서 크게 감소했는지 확인
        assert!(final_val < 1.0, "y should decrease from 401, got {}", final_val);
    }
}
