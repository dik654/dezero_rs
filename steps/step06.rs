use ndarray::ArrayD;

struct Variable {
    data: ArrayD<f64>,
    /// 역전파 시 기울기(미분값) 저장
    grad: Option<ArrayD<f64>>,
}

impl Variable {
    fn new(data: ArrayD<f64>) -> Self {
        Variable { data, grad: None }
    }
}

trait Function {
    fn forward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&self, gy: &ArrayD<f64>) -> ArrayD<f64>;

    /// Python에서는 __call__ 안에서 self.input = input으로 입력을 저장하지만
    /// Rust의 trait에는 필드가 없으므로 각 구현체의 forward에서 입력을 저장한다
    fn call(&mut self, input: &Variable) -> Variable {
        let y = self.forward(&input.data);
        Variable::new(y)
    }
}

/// 제곱 연산
struct Square {
    /// Python의 self.input에 해당
    /// forward에서 저장하고 backward에서 사용
    input: Option<ArrayD<f64>>,
}

impl Square {
    fn new() -> Self {
        Square { input: None }
    }
}

impl Function for Square {
    fn forward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        // Python에서는 __call__에서 self.input = input 하지만
        // Rust에서는 trait에 필드가 없으므로 여기서 저장
        self.input = Some(x.clone());
        x * x
    }

    fn backward(&self, gy: &ArrayD<f64>) -> ArrayD<f64> {
        let x = self.input.as_ref().unwrap();
        // gx = 2 * x * gy
        // 미분값 2x와 출력쪽에서 전해지는 미분값(gy) 전달
        2.0 * x * gy
    }
}

/// 지수 연산
struct Exp {
    input: Option<ArrayD<f64>>,
}

impl Exp {
    fn new() -> Self {
        Exp { input: None }
    }
}

impl Function for Exp {
    fn forward(&mut self, x: &ArrayD<f64>) -> ArrayD<f64> {
        self.input = Some(x.clone());
        x.mapv(f64::exp)
    }

    fn backward(&self, gy: &ArrayD<f64>) -> ArrayD<f64> {
        let x = self.input.as_ref().unwrap();
        // 미분값 exp^x와 출력쪽에서 전해지는 미분값(gy) 전달
        x.mapv(f64::exp) * gy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backprop_manual() {
        let mut a = Square::new();
        let mut b = Exp::new();
        let mut c = Square::new();

        // 순전파: x -> Square -> Exp -> Square -> y
        // 차례대로 호출
        let mut x = Variable::new(ndarray::arr0(0.5).into_dyn());
        let mut a_out = a.call(&x);
        let mut b_out = b.call(&a_out);
        let mut y = c.call(&b_out);

        // 역전파: 각 Variable의 grad에 기울기를 저장
        // dy/dx를 구한다
        // y.grad     = dy/dy = 1.0          ← 출발점
        // b_out.grad = dy/db_out            ← y가 b_out에 대해
        // a_out.grad = dy/da_out            ← y가 a_out에 대해
        // x.grad     = dy/dx                ← y가 x에 대해

        y.grad = Some(ndarray::arr0(1.0).into_dyn());
        b_out.grad = Some(c.backward(y.grad.as_ref().unwrap()));
        a_out.grad = Some(b.backward(b_out.grad.as_ref().unwrap()));
        x.grad = Some(a.backward(a_out.grad.as_ref().unwrap()));

        // f(x) = (e^(x^2))^2 = e^(2x^2), f'(x) = 4x * e^(2x^2)
        let expected = 4.0 * 0.5 * (2.0 * 0.5_f64.powi(2)).exp();
        let result = x.grad.as_ref().unwrap().first().unwrap();
        assert!((result - expected).abs() < 1e-10);
    }
}
