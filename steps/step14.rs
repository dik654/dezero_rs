use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

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

    /// 기울기 초기화
    /// 같은 변수로 다른 계산을 할 때 이전 기울기를 리셋
    fn cleargrad(&self) {
        self.inner.borrow_mut().grad = None;
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
            let (gxs, inputs) = {
                let state = state_ref.borrow();
                let gys: Vec<ArrayD<f64>> = state
                    .outputs
                    .iter()
                    .map(|o| o.inner.borrow().grad.clone().unwrap())
                    .collect();
                let xs: Vec<ArrayD<f64>> = state
                    .inputs
                    .iter()
                    .map(|i| i.inner.borrow().data.clone())
                    .collect();
                let inputs = state.inputs.clone();
                let gxs = state.func.backward(&xs, &gys);
                (gxs, inputs)
            };
            for (input, gx) in inputs.iter().zip(gxs) {
                // 덮어쓰기 → 누적
                let mut inner = input.inner.borrow_mut();
                if inner.grad.is_none() {
                    inner.grad = Some(gx);
                } else {
                    // 같은 변수가 여러 경로에서 사용되면 기울기를 더한다
                    inner.grad = Some(inner.grad.as_ref().unwrap() + &gx);
                }
                drop(inner);
                if let Some(creator) = input.inner.borrow().creator.clone() {
                    funcs.push(creator);
                }
            }
        }
    }
}

trait Function {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
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
        outputs.into_iter().next().unwrap()
    }
}

struct Add;

impl Add {
    fn new() -> Func {
        Func::new(Add)
    }
}

impl Function for Add {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

fn add(x0: &Variable, x1: &Variable) -> Variable {
    Add::new().call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_variable_add() {
        // y = x + x = 2x, dy/dx = 2
        let x = Variable::new(ndarray::arr0(3.0).into_dyn());
        let y = add(&x, &x);
        y.backward();

        let x_grad = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        // 누적이라 2.0
        assert_eq!(x_grad, 2.0);
    }

    #[test]
    fn test_same_variable_triple_add() {
        // y = (x + x) + x = 3x, dy/dx = 3
        let x = Variable::new(ndarray::arr0(3.0).into_dyn());
        let y = add(&add(&x, &x), &x);
        y.backward();

        let x_grad = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert_eq!(x_grad, 3.0);
    }

    #[test]
    fn test_cleargrad() {
        // 1회차: y = x + x, dy/dx = 2
        let x = Variable::new(ndarray::arr0(3.0).into_dyn());
        let y = add(&x, &x);
        y.backward();
        let grad1 = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert_eq!(grad1, 2.0);

        // cleargrad 없이 다시 backward하면 이전 기울기가 남아있음
        // cleargrad로 초기화 후 새로 계산
        x.cleargrad();

        // 2회차: y = (x + x) + x, dy/dx = 3
        let y = add(&add(&x, &x), &x);
        y.backward();
        let grad2 = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert_eq!(grad2, 3.0);
    }
}
