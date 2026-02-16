// step29: 뉴턴법을 이용한 최적화
// 경사하강법과 달리 2차 도함수(f'')를 사용하여 lr 없이 빠르게 수렴한다
// 갱신 규칙: x = x - f'(x) / f''(x)

use dezero::Variable;
use ndarray::ArrayD;

fn f(x: &Variable) -> Variable {
    // y = x^4 - 2x^2
    let t1 = x.pow(4.0);
    let t2 = 2.0 * &x.pow(2.0);
    &t1 - &t2
}

/// 2차 도함수 (수동 계산)
/// f(x) = x^4 - 2x^2
/// f'(x) = 4x^3 - 4x
/// f''(x) = 12x^2 - 4
fn gx2(x: &ArrayD<f64>) -> ArrayD<f64> {
    12.0 * x.mapv(|v| v * v) - 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_method() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let iters = 10;

        for i in 0..iters {
            println!("{} x={:.6}", i, x.data().iter().next().unwrap());

            let y = f(&x);
            x.cleargrad();
            y.backward(false, false);

            let grad = x.grad().unwrap();
            x.set_data(x.data() - &grad / &gx2(&x.data()));
        }

        // f(x) = x^4 - 2x^2 の最솟값은 x = ±1에서 y = -1
        let final_x = *x.data().iter().next().unwrap();
        println!("final: x={:.10}", final_x);

        assert!(
            (final_x.abs() - 1.0).abs() < 1e-6,
            "x should converge to ±1, got {}",
            final_x
        );
    }
}
