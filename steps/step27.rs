// step27: 테일러 급수로 sin 함수의 미분 (역전파)
// 내장 Sin 함수와 테일러 급수 근사의 결과를 비교한다

use dezero::{plot_dot_graph, sin, Variable};

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, i| acc * i as f64)
}

/// 테일러 급수로 sin(x)를 근사하는 함수
/// sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
fn my_sin(x: &Variable, threshold: f64) -> Variable {
    let mut y = Variable::new(ndarray::arr0(0.0).into_dyn());
    for i in 0..100000 {
        // 숫자
        let c = (-1.0_f64).powi(i as i32) / factorial(2 * i + 1);
        // 숫자 * 변수 x
        let t = c * &x.pow((2 * i + 1) as f64);
        y = &y + &t;
        if t.data().iter().next().unwrap().abs() < threshold {
            break;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 내장 sin 함수의 forward/backward 검증
    #[test]
    fn test_sin() {
        let x = Variable::new(ndarray::arr0(std::f64::consts::FRAC_PI_4).into_dyn());
        let y = sin(&x);
        y.backward(false);

        println!("--- original sin ---");
        println!("{:?}", y.data());
        println!("{:?}", x.grad().unwrap());

        let expected_y = (std::f64::consts::FRAC_PI_4).sin();
        let expected_grad = (std::f64::consts::FRAC_PI_4).cos();
        assert!((y.data().iter().next().unwrap() - expected_y).abs() < 1e-10);
        assert!((x.grad().unwrap().iter().next().unwrap() - expected_grad).abs() < 1e-10);
    }

    /// 테일러 급수 근사 sin의 forward/backward 검증
    #[test]
    fn test_my_sin() {
        let x = Variable::new(ndarray::arr0(std::f64::consts::FRAC_PI_4).into_dyn());
        let y = my_sin(&x, 0.0001);
        y.backward(false);

        println!("--- approximate sin ---");
        println!("{:?}", y.data());
        println!("{:?}", x.grad().unwrap());

        let expected_y = (std::f64::consts::FRAC_PI_4).sin();
        let expected_grad = (std::f64::consts::FRAC_PI_4).cos();
        assert!((y.data().iter().next().unwrap() - expected_y).abs() < 1e-4);
        assert!((x.grad().unwrap().iter().next().unwrap() - expected_grad).abs() < 1e-4);
    }

    /// 계산 그래프 시각화 (Graphviz 설치 시)
    #[test]
    fn test_plot_my_sin() {
        let x = Variable::new(ndarray::arr0(std::f64::consts::FRAC_PI_4).into_dyn());
        let y = my_sin(&x, 0.0001);
        y.backward(false);

        x.set_name("x");
        y.set_name("y");

        let result = plot_dot_graph(&y, false, "my_sin.png");
        if result.is_ok() {
            assert!(std::path::Path::new("my_sin.png").exists());
            let _ = std::fs::remove_file("my_sin.png");
            let _ = std::fs::remove_file("my_sin.dot");
        }
    }
}
