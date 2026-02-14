use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// Function의 상태를 담는 구조체
struct FuncState {
    func: Box<dyn Function>,
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

trait Function {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64>;
}

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

    /// step12에서 변경: 단일 출력이면 Variable을 직접 반환
    /// Python의 return outputs if len(outputs) > 1 else outputs[0] 에 해당
    fn call(&self, inputs: &[&Variable]) -> Variable {
        let xs: Vec<ArrayD<f64>> = inputs.iter().map(|v| v.inner.borrow().data.clone()).collect();
        let ys = {
            let mut state = self.state.borrow_mut();
            state.inputs = inputs.iter().map(|v| (*v).clone()).collect();
            state.func.forward(&xs)
        };
        let outputs: Vec<Variable> = ys.into_iter().map(Variable::new).collect();
        for output in &outputs {
            output.set_creator(&self.state);
        }
        self.state.borrow_mut().outputs = outputs.clone();
        // 단일 출력 반환 (Python의 outputs[0])
        outputs.into_iter().next().unwrap()
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
        vec![x0 + x1]
    }

    fn backward(&self, _x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        gy.clone()
    }
}

/// step12에서 변경: 편의 함수도 Variable을 직접 반환 (call이 Variable을 반환하므로 [0].clone() 불필요)
fn square(x: &Variable) -> Variable {
    Square::new().call(&[x])
}

fn exp(x: &Variable) -> Variable {
    Exp::new().call(&[x])
}

/// add 편의 함수 추가
/// 인수로 Variable 직접 받아 처리
fn add(x0: &Variable, x1: &Variable) -> Variable {
    Add::new().call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_forward() {
        let x0 = Variable::new(ndarray::arr0(2.0).into_dyn());
        let x1 = Variable::new(ndarray::arr0(3.0).into_dyn());
        // call이 Variable을 직접 반환 (Vec로 처리 x)
        let y = add(&x0, &x1);
        let result = *y.inner.borrow().data.first().unwrap();
        assert_eq!(result, 5.0);
    }

}
