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

/// 지수(e^x) 연산을 나타내는 구조체
struct Exp;

impl Function for Exp {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        // mapv: 배열의 각 원소에 함수를 적용한 새 배열을 반환 (numpy의 np.exp(x)에 해당)
        // f64::exp: 표준 라이브러리의 e^x 함수
        // [1, 2, 3].mapv(f64::exp) -> [e^1, e^2, e^3]
        x.mapv(f64::exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_composition() {
        // 함수 합성: Square -> Exp -> Square
        let a = Square;
        let b = Exp;
        let c = Square;

        let x = Variable::new(ndarray::arr0(0.5).into_dyn());
        let a_out = a.call(&x);
        let b_out = b.call(&a_out);
        let y = c.call(&b_out);

        // 0.5 -> 0.25 -> e^0.25 -> (e^0.25)^2
        let expected = (0.25_f64.exp()).powi(2);
        let result = y.data.first().unwrap();
        // 부동 소수점은 근사값이기에 ==는 미세한 오차로 실패할 수 있어
        // 차이가 미세한지 여부로 검사
        assert!((result - expected).abs() < 1e-10);
    }
}
