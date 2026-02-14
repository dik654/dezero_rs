use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// Function의 상태를 담는 구조체
/// 단일(Option<Variable>)에서 -> 복수(iVec<Variable>)로 변경
struct FuncState {
    func: Box<dyn Function>,
    // 단일 → 복수로 변경
    inputs: Vec<Variable>,
    outputs: Vec<Variable>,
}

type FuncStateRef = Rc<RefCell<FuncState>>;

struct VarInner {
    data: ArrayD<f64>,
    grad: Option<ArrayD<f64>>,
    creator: Option<FuncStateRef>,
}

#[derive(Clone)]
struct Variable {
    inner: Rc<RefCell<VarInner>>,
}

impl Variable {
    fn new(data: ArrayD<f64>) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VarInner {
                data,
                grad: None,
                creator: None,
            })),
        }
    }

    fn set_creator(&self, state: &FuncStateRef) {
        self.inner.borrow_mut().creator = Some(Rc::clone(state));
    }

    fn backward(&self) {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(ArrayD::ones(inner.data.shape()));
            }
        }

        let mut funcs: Vec<FuncStateRef> = Vec::new();
        if let Some(creator) = self.inner.borrow().creator.clone() {
            funcs.push(creator);
        }

        while let Some(state_ref) = funcs.pop() {
            let (gx, input) = {
                let state = state_ref.borrow();
                // 아직 단일 입출력 기준으로 동작 (다중 입출력 backward는 이후 스텝에서 구현)
                let output = &state.outputs[0];
                let gy = output.inner.borrow().grad.clone().unwrap();
                let input = state.inputs[0].clone();
                let x = input.inner.borrow().data.clone();
                let gx = state.func.backward(&x, &gy);
                (gx, input)
            };
            input.inner.borrow_mut().grad = Some(gx);
            if let Some(creator) = input.inner.borrow().creator.clone() {
                funcs.push(creator);
            }
        }
    }
}

/// step11에서 forward가 복수 입력/출력으로 변경
trait Function {
    /// 복수 입력(&[ArrayD<f64>]) → 복수 출력(Vec<ArrayD<f64>>)
    /// Vec는 소유권을 가져가기 때문에 빌려서 읽기만 하기 위해서 입력은 슬라이스 사용(&[])
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    /// backward는 아직 단일 입출력
    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64>;
}

/// Function의 newtype 래퍼
struct Func {
    state: FuncStateRef,
}

impl Func {
    fn new(func: impl Function + 'static) -> Self {
        Func {
            state: Rc::new(RefCell::new(FuncState {
                func: Box::new(func),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })),
        }
    }

    /// 복수 입력을 받아 복수 출력을 반환
    fn call(&self, inputs: &[&Variable]) -> Vec<Variable> {
        // 각 Variable에서 데이터를 꺼내기 (Python의 xs = [x.data for x in inputs])
        let xs: Vec<ArrayD<f64>> = inputs.iter().map(|v| v.inner.borrow().data.clone()).collect();
        let ys = {
            let mut state = self.state.borrow_mut();
            state.inputs = inputs.iter().map(|v| (*v).clone()).collect();
            state.func.forward(&xs)
        };
        // 출력을 Variable로 감싸기 (Python의 outputs = [Variable(as_array(y)) for y in ys])
        let outputs: Vec<Variable> = ys.into_iter().map(Variable::new).collect();
        for output in &outputs {
            output.set_creator(&self.state);
        }
        self.state.borrow_mut().outputs = outputs.clone();
        outputs
    }
}

/// 제곱 연산
struct Square;

impl Square {
    fn new() -> Func {
        Func::new(Square)
    }
}

impl Function for Square {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        vec![x * x]
    }

    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        2.0 * x * gy
    }
}

/// 지수 연산
struct Exp;

impl Exp {
    fn new() -> Func {
        Func::new(Exp)
    }
}

impl Function for Exp {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        vec![x.mapv(f64::exp)]
    }

    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        x.mapv(f64::exp) * gy
    }
}

/// 덧셈 연산
struct Add;

impl Add {
    fn new() -> Func {
        Func::new(Add)
    }
}

impl Function for Add {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = (&xs[0], &xs[1]);
        // 단일 출력 벡터 (두 입력을 더해서 하나의 출력으로)
        vec![x0 + x1]
    }

    fn backward(&self, _x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        gy.clone()
    }
}

/// 편의 함수
fn square(x: &Variable) -> Variable {
    Square::new().call(&[x])[0].clone()
}

fn exp(x: &Variable) -> Variable {
    Exp::new().call(&[x])[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        // 덧셈 잘 되는지 테스트
        let x0 = Variable::new(ndarray::arr0(2.0).into_dyn());
        let x1 = Variable::new(ndarray::arr0(3.0).into_dyn());
        let ys = Add::new().call(&[&x0, &x1]);
        let y = &ys[0];
        let result = *y.inner.borrow().data.first().unwrap();
        // 정수의 합이므로 정확하게 비교 가능
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_square_exp_backward() {
        // 기존 단일 입출력 backward도 정상 동작 확인
        let x = Variable::new(ndarray::arr0(0.5).into_dyn());
        let y = square(&exp(&square(&x)));
        y.backward();

        let expected = 4.0 * 0.5 * (2.0 * 0.5_f64.powi(2)).exp();
        let result = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert!((result - expected).abs() < 1e-10);
    }
}
