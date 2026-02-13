use ndarray::ArrayD;

struct Variable {
    data: ArrayD<f64>,
}

impl Variable {
    fn new(data: ArrayD<f64>) -> Self {
        Variable { data }
    }
}

trait Function {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64>;

    fn call(&self, input: &Variable) -> Variable {
        let y = self.forward(&input.data);
        Variable::new(y)
    }
}

struct Square;

impl Function for Square {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        x * x
    }
}

/// 수치 미분(numerical differentiation): 중앙 차분법으로 도함수를 근사
/// f'(x) ≈ (f(x+h) - f(x-h)) / 2h
fn numerical_diff(f: &dyn Function, x: &Variable, eps: f64) -> ArrayD<f64> {
    // x-h
    let x0 = Variable::new(&x.data - eps);
    // x+h
    let x1 = Variable::new(&x.data + eps);
    // f(x-h)
    let y0 = f.call(&x0);
    // f(x+h)
    let y1 = f.call(&x1);
    // (f(x+h) - f(x-h)) / 2h
    (y1.data - y0.data) / (2.0 * eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_diff_square() {
        let f = Square;
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        // f(x) = x^2 의 도함수는 f'(x) = 2x, x=2일 때 4.0
        let dy = numerical_diff(&f, &x, 1e-4);
        // 배열에서 f64 값 꺼내기
        let result = dy.first().unwrap();
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_diff_composition() {
        // 합성 함수의 수치 미분
        // f(x) = (e^(x^2))^2 의 도함수를 수치적으로 구함
        let x = Variable::new(ndarray::arr0(0.5).into_dyn());

        // 합성 함수를 하나의 Function으로 감싸기
        struct ComposedFn;
        impl Function for ComposedFn {
            fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
                let a = x * x;             // Square
                let b = a.mapv(f64::exp);  // Exp
                &b * &b                    // Square
            }
        }

        // 합성함수 결과 (exp^x^2)^2
        let f = ComposedFn;
        // 합성함수 미분
        let dy = numerical_diff(&f, &x, 1e-4);

        // f(x) = e^(2x^2), f'(x) = 4x * e^(2x^2)
        let expected = 4.0 * 0.5 * (2.0 * 0.5_f64.powi(2)).exp();
        let result = dy.first().unwrap();
        assert!((result - expected).abs() < 1e-4);
    }
}
