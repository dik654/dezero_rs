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

    /// 다중 입출력 backward
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
            // 모든 출력의 grad를 모은다 (Python의 gys = [output.grad for output in f.outputs])
            // 모든 입력의 데이터를 모은다
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
                // 다중 입력 grad를 한 번에 계산 (Python의 gxs = f.backward(*gys))
                let gxs = state.func.backward(&xs, &gys);
                (gxs, inputs)
            };
            // 각 입력 변수에 grad를 저장 (Python의 for x, gx in zip(f.inputs, gxs))
            for (input, gx) in inputs.iter().zip(gxs) {
                input.inner.borrow_mut().grad = Some(gx);
                if let Some(creator) = input.inner.borrow().creator.clone() {
                    funcs.push(creator);
                }
            }
        }
    }
}

/// backward도 다중 입출력
trait Function {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    /// 기존 &ArrayD<f64>, &ArrayD<f64> -> ArrayD<f64>
    /// 수정 &[ArrayD<f64>], &[ArrayD<f64>] → Vec<ArrayD<f64>>
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

/// 제곱 연산
struct Square;

impl Square {
    fn new() -> Func {
        Func::new(Square)
    }
}

impl Function for Square {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[0]]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        // gx = 2 * x * gy
        vec![2.0 * &xs[0] * &gys[0]]
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
        vec![xs[0].mapv(f64::exp)]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::exp) * &gys[0]]
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
        vec![&xs[0] + &xs[1]]
    }

    /// 덧셈의 역전파: 출력 기울기가 양쪽 입력에 그대로 전달 (return gy, gy)
    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

fn square(x: &Variable) -> Variable {
    Square::new().call(&[x])
}

fn add(x0: &Variable, x1: &Variable) -> Variable {
    Add::new().call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_square_backward() {
        // z = add(square(x), square(y))
        // z = x^2 + y^2
        // dz/dx = 2x = 2*2 = 4
        // dz/dy = 2y = 2*3 = 6
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let y = Variable::new(ndarray::arr0(3.0).into_dyn());

        let z = add(&square(&x), &square(&y));
        // 한 번의 역전파 실행으로 x, y의 기울기가 둘 다 계산 
        z.backward();

        let z_val = *z.inner.borrow().data.first().unwrap();
        let x_grad = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        let y_grad = *y.inner.borrow().grad.as_ref().unwrap().first().unwrap();

        // z = 2^2 + 3^2 = 13
        assert_eq!(z_val, 13.0);
        // dz/dx = 2*2 = 4
        assert_eq!(x_grad, 4.0);
        // dz/dy = 2*3 = 6
        assert_eq!(y_grad, 6.0);
    }
}
