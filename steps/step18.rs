// step18: 메모리 절약 모드
// 기존(step17)에는 순전파 시 계산 그래프(creator, inputs, outputs)를 항상 생성했다
// 하지만 추론(예측)할 때는 역전파를 하지 않으므로 그래프가 필요 없다
// enable_backprop 플래그로 그래프 생성 여부를 제어하고
// no_grad() 가드로 스코프 안에서만 그래프 생성을 끌 수 있게 한다

use ndarray::ArrayD;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::{Rc, Weak};

// 전역 설정: 역전파 그래프를 만들지 여부
// thread_local!로 스레드별 전역 변수를 만든다
// Cell은 Copy 타입(bool)에 대한 내부 가변성 (RefCell보다 가벼움)
thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
}

/// no_grad 스코프 가드
/// Python의 `with no_grad():` 에 해당
/// 생성 시 enable_backprop을 false로, Drop 시 원래 값으로 복원
struct NoGradGuard {
    prev: bool,
}

/// 이 스코프 안에서는 계산 그래프를 만들지 않는다 (순전파만)
fn no_grad() -> NoGradGuard {
    let prev = ENABLE_BACKPROP.with(|c| c.get());
    ENABLE_BACKPROP.with(|c| c.set(false));
    NoGradGuard { prev }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        // 스코프를 벗어나면 원래 값으로 복원
        ENABLE_BACKPROP.with(|c| c.set(self.prev));
    }
}

// --- 핵심 구조체 ---

struct FuncState {
    func: Box<dyn Function>,
    generation: u32,
    inputs: Vec<Variable>,
    outputs: Vec<Weak<RefCell<VarInner>>>,
}

type FuncStateRef = Rc<RefCell<FuncState>>;

struct VarInner {
    data: ArrayD<f64>,
    grad: Option<ArrayD<f64>>,
    creator: Option<FuncStateRef>,
    generation: u32,
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
                generation: 0,
            })),
        }
    }

    fn set_creator(&self, state: &FuncStateRef) {
        let func_gen = state.borrow().generation;
        let mut inner = self.inner.borrow_mut();
        inner.creator = Some(Rc::clone(state));
        inner.generation = func_gen + 1;
    }

    fn cleargrad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    fn backward(&self, retain_grad: bool) {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(ArrayD::ones(inner.data.shape()));
            }
        }

        let mut funcs: Vec<FuncStateRef> = Vec::new();
        let mut seen: HashSet<*const RefCell<FuncState>> = HashSet::new();

        let add_func = |f: FuncStateRef, funcs: &mut Vec<FuncStateRef>, seen: &mut HashSet<*const RefCell<FuncState>>| {
            let ptr = Rc::as_ptr(&f);
            if !seen.contains(&ptr) {
                seen.insert(ptr);
                funcs.push(f);
                funcs.sort_by_key(|f| f.borrow().generation);
            }
        };

        if let Some(creator) = self.inner.borrow().creator.clone() {
            add_func(creator, &mut funcs, &mut seen);
        }

        while let Some(state_ref) = funcs.pop() {
            let (gxs, inputs) = {
                let state = state_ref.borrow();
                let gys: Vec<ArrayD<f64>> = state
                    .outputs
                    .iter()
                    .map(|o| o.upgrade().unwrap().borrow().grad.clone().unwrap())
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

            if !retain_grad {
                let state = state_ref.borrow();
                for output in &state.outputs {
                    if let Some(out) = output.upgrade() {
                        out.borrow_mut().grad = None;
                    }
                }
            }

            for (input, gx) in inputs.iter().zip(gxs) {
                let mut inner = input.inner.borrow_mut();
                if inner.grad.is_none() {
                    inner.grad = Some(gx);
                } else {
                    inner.grad = Some(inner.grad.as_ref().unwrap() + &gx);
                }
                drop(inner);
                if let Some(creator) = input.inner.borrow().creator.clone() {
                    add_func(creator, &mut funcs, &mut seen);
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
                generation: 0,
                inputs: Vec::new(),
                outputs: Vec::new(),
            })),
        }
    }

    fn call(&self, inputs: &[&Variable]) -> Variable {
        let xs: Vec<ArrayD<f64>> = inputs.iter().map(|v| v.inner.borrow().data.clone()).collect();

        let ys = self.state.borrow().func.forward(&xs);
        let outputs: Vec<Variable> = ys.into_iter().map(Variable::new).collect();

        // enable_backprop이 true일 때만 계산 그래프를 만든다
        // false면 순전파 결과만 반환 (추론 모드)
        if ENABLE_BACKPROP.with(|c| c.get()) {
            let max_gen = inputs
                .iter()
                .map(|v| v.inner.borrow().generation)
                .max()
                .unwrap_or(0);

            {
                let mut state = self.state.borrow_mut();
                state.inputs = inputs.iter().map(|v| (*v).clone()).collect();
                state.generation = max_gen;
            }
            for output in &outputs {
                output.set_creator(&self.state);
            }
            self.state.borrow_mut().outputs = outputs
                .iter()
                .map(|o| Rc::downgrade(&o.inner))
                .collect();
        }

        outputs.into_iter().next().unwrap()
    }
}

// --- 함수 구현 ---

struct Square;

impl Function for Square {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[0]]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![2.0 * &xs[0] * &gys[0]]
    }
}

struct Add;

impl Function for Add {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

fn square(x: &Variable) -> Variable {
    Func::new(Square).call(&[x])
}

fn add(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(Add).call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 역전파 정상 동작 (enable_backprop = true, 기본)
    #[test]
    fn test_with_backprop() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let y = square(&x);
        y.backward(false);

        assert_eq!(
            *x.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            4.0 // dy/dx = 2x = 4
        );
    }

    /// no_grad 스코프: 계산 그래프를 만들지 않는다
    #[test]
    fn test_no_grad() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let y;
        {
            let _guard = no_grad(); // enable_backprop = false
            y = square(&x);
        }
        // no_grad 안에서 만든 y는 creator가 없다
        assert!(y.inner.borrow().creator.is_none());

        // 스코프를 벗어나면 enable_backprop이 true로 복원된다
        let z = square(&x);
        assert!(z.inner.borrow().creator.is_some());
    }

    /// no_grad 안에서는 순전파 결과는 정상
    #[test]
    fn test_no_grad_forward_works() {
        let x = Variable::new(ndarray::arr0(3.0).into_dyn());
        let y = {
            let _guard = no_grad();
            add(&square(&x), &square(&x))
        };
        // 순전파 결과: 3² + 3² = 18
        assert_eq!(
            *y.inner.borrow().data.first().unwrap(),
            18.0
        );
    }
}
