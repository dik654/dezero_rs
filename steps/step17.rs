use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

struct FuncState {
    func: Box<dyn Function>,
    generation: u32,
    inputs: Vec<Variable>,
    // Weak 참조로 출력 변수를 저장한다
    // 만약 Rc로 저장하면:
    //   Variable ──(Rc)──→ FuncState ──(Rc)──→ Variable
    //     (creator)                     (outputs)
    //   서로 Rc로 참조 → 참조 카운트가 영원히 1 이상 → 메모리 누수
    //
    // Weak은 참조 카운트를 올리지 않으므로 순환이 끊긴다:
    //   Variable ──(Rc)──→ FuncState ──(Weak)──→ Variable
    //   Variable이 사라지면 참조 카운트 0 → 정상 해제
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

    /// retain_grad: false(기본) → 중간 변수의 grad 삭제, 입력 변수의 grad만 유지
    ///             true(디버깅) → 모든 변수의 grad 유지 (기존 step16과 동일 동작)
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
                // Weak → upgrade()로 Rc를 복원한다
                // upgrade()는 대상이 살아있으면 Some(Rc), 해제됐으면 None
                // 역전파 시점에는 출력 변수가 살아있으므로 unwrap() 안전
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

            // retain_grad가 false(기본)면 중간 변수의 grad를 삭제해서 메모리 절약
            // 중간 grad는 역전파 계산에 이미 사용했으므로 더 이상 필요 없다
            // true로 하면 디버깅용으로 중간 변수의 grad를 확인할 수 있다
            if !retain_grad {
                let state = state_ref.borrow();
                for output in &state.outputs {
                    if let Some(out) = output.upgrade() {
                        // 역전파 계산에 이미 사용된 중간 변수의 기울기값만 제거
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

        let max_gen = inputs
            .iter()
            .map(|v| v.inner.borrow().generation)
            .max()
            .unwrap_or(0);

        let ys = {
            let mut state = self.state.borrow_mut();
            state.inputs = inputs.iter().map(|v| (*v).clone()).collect();
            state.generation = max_gen;
            state.func.forward(&xs)
        };
        let outputs: Vec<Variable> = ys.into_iter().map(Variable::new).collect();
        for output in &outputs {
            output.set_creator(&self.state);
        }
        // Rc::downgrade()로 Weak 참조를 만들어 저장
        // Rc → Weak: 참조 카운트를 올리지 않는 포인터
        // 나중에 접근할 때는 upgrade()로 Rc를 복원해야 한다
        self.state.borrow_mut().outputs = outputs
            .iter()
            .map(|o| Rc::downgrade(&o.inner))
            .collect();
        outputs.into_iter().next().unwrap()
    }
}

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

    /// retain_grad=false: 중간 변수의 grad는 삭제됨
    #[test]
    fn test_no_retain_grad() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward(false);

        // 최종 입력 x의 grad는 유지
        assert_eq!(
            *x.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            64.0
        );
        // 중간 변수 a의 grad는 삭제됨
        assert!(a.inner.borrow().grad.is_none());
    }

    /// retain_grad=true: 중간 변수의 grad도 유지
    #[test]
    fn test_retain_grad() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward(true);

        assert_eq!(
            *x.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            64.0
        );
        // retain_grad=true이므로 중간 변수 a의 grad도 유지
        assert!(a.inner.borrow().grad.is_some());
        // a의 grad = 2a + 2a = 4a = 16
        assert_eq!(
            *a.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            16.0
        );
    }
}
