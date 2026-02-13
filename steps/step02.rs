use ndarray::ArrayD;

struct Variable {
    data: ArrayD<f64>,
}

impl Variable {
    fn new(data: ArrayD<f64>) -> Self {
        Variable { data }
    }
}

/// 타입이 할 수 있는 행동(can do)을 정의하는 trait (go, java의 interface와 유사)
trait Function {
    /// 순전파 연산을 선언만하고 구현체(impl)에서 실제 로직을 정의
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64>;

    /// 기본 구현(default implementation): 순전파를 호출하고 결과를 Variable로 감싸 반환한다
    /// 구현체에서 재정의(override) 안해도 그대로 사용 가능
    fn call(&self, input: &Variable) -> Variable {
        let y = self.forward(&input.data);
        Variable::new(y)
    }
}

/// 제곱 연산을 나타내는 구조체로 Function trait을 구현
struct Square;

impl Function for Square {
    /// Function::forward의 실제 구현으로 각 원소를 제곱한 배열을 반환
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        x * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_forward() {
        let x = Variable::new(ndarray::arr0(10.0).into_dyn());
        let f = Square;
        // 파이썬에서는 __call__을 통해 인스턴스를 함수처럼 호출하지만 
        // stable rust에는 없으므로 명시적으로 실행
        let y = f.call(&x);
        assert_eq!(y.data, ndarray::arr0(100.0).into_dyn());
    }
}
