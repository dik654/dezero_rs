// step33: 뉴턴법을 이용한 최적화 (자동 2차 미분)
// step29에서는 f''(x)를 수동으로 계산했지만
// 여기서는 backward(create_graph=True)로 역전파 그래프를 만들어
// 2번째 backward로 자동으로 f''(x)를 계산한다

use dezero::Variable;

fn f(x: &Variable) -> Variable {
    // y = x^4 - 2x^2
    let t1 = x.pow(4.0);
    let t2 = 2.0 * &x.pow(2.0);
    &t1 - &t2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_backprop_newton() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let iters = 10;

        for i in 0..iters {
            println!("{} x={:.6}", i, x.data().iter().next().unwrap());

            let y = f(&x);
            x.cleargrad();
            // 1차 역전파: create_graph=true로 역전파 계산 자체도 그래프에 기록
            y.backward(false, true);

            // gx는 Variable (그래프 연결 정보를 가지고 있음)
            let gx = x.grad_var().unwrap();
            x.cleargrad();
            // 2차 역전파: gx를 역전파하여 f''(x) 계산
            gx.backward(false, false);

            let gx2 = x.grad().unwrap();
            let gx_data = gx.data();
            // 뉴턴법 갱신: x = x - f'(x) / f''(x)
            x.set_data(x.data() - &gx_data / &gx2);
        }

        // f(x) = x^4 - 2x^2 의 최솟값은 x = ±1에서 y = -1
        let final_x = *x.data().iter().next().unwrap();
        println!("final: x={:.10}", final_x);

        assert!(
            (final_x.abs() - 1.0).abs() < 1e-6,
            "x should converge to ±1, got {}",
            final_x
        );
    }
}
