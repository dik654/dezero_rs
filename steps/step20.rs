// step20: 연산자 오버로딩
// add(&x, &y) 대신 &x + &y, mul(&x, &y) 대신 &x * &y 로 쓸 수 있게 한다
// Mul 함수를 추가하고, std::ops::Add와 std::ops::Mul 트레잇을 구현한다

use ndarray::ArrayD;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt;
use std::rc::{Rc, Weak};

thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
}

struct NoGradGuard {
    prev: bool,
}

fn no_grad() -> NoGradGuard {
    let prev = ENABLE_BACKPROP.with(|c| c.get());
    ENABLE_BACKPROP.with(|c| c.set(false));
    NoGradGuard { prev }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
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
    name: Option<String>,
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
                name: None,
            })),
        }
    }

    fn with_name(data: ArrayD<f64>, name: &str) -> Self {
        Variable {
            inner: Rc::new(RefCell::new(VarInner {
                data,
                grad: None,
                creator: None,
                generation: 0,
                name: Some(name.to_string()),
            })),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    fn ndim(&self) -> usize {
        self.inner.borrow().data.ndim()
    }

    fn size(&self) -> usize {
        self.inner.borrow().data.len()
    }

    fn len(&self) -> usize {
        self.inner.borrow().data.shape().first().copied().unwrap_or(0)
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

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        let name = inner.name.as_deref().unwrap_or("");
        let data_str = format!("{}", inner.data);
        if name.is_empty() {
            write!(f, "variable({})", data_str)
        } else {
            write!(f, "variable({}, name={})", data_str, name)
        }
    }
}

// --- 연산자 오버로딩 ---
// Python의 __add__, __mul__ 에 해당
// &Variable 간의 연산을 지원한다 (Rc clone이므로 가벼움)

/// &x + &y → add(&x, &y)
impl std::ops::Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: Self) -> Variable {
        add(self, rhs)
    }
}

/// &x * &y → mul(&x, &y)
impl std::ops::Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: Self) -> Variable {
        mul(self, rhs)
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

/// 덧셈 함수 (Function 트레잇 구현용, std::ops::Add와 이름 충돌 없음)
struct AddFn;

impl Function for AddFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

/// 곱셈 함수
/// forward: y = x0 * x1
/// backward: dy/dx0 = x1, dy/dx1 = x0 (곱의 미분)
struct MulFn;

impl Function for MulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[1]]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        // dy/dx0 = x1 * gy, dy/dx1 = x0 * gy
        vec![&xs[1] * &gys[0], &xs[0] * &gys[0]]
    }
}

fn square(x: &Variable) -> Variable {
    Func::new(Square).call(&[x])
}

fn add(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(AddFn).call(&[x0, x1])
}

fn mul(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(MulFn).call(&[x0, x1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    /// 곱셈 순전파
    #[test]
    fn test_mul_forward() {
        let a = Variable::new(arr0(3.0).into_dyn());
        let b = Variable::new(arr0(2.0).into_dyn());
        let y = mul(&a, &b);
        assert_eq!(*y.inner.borrow().data.first().unwrap(), 6.0);
    }

    /// 곱셈 역전파
    #[test]
    fn test_mul_backward() {
        let a = Variable::new(arr0(3.0).into_dyn());
        let b = Variable::new(arr0(2.0).into_dyn());
        let y = mul(&a, &b);
        y.backward(false);

        // dy/da = b = 2.0
        assert_eq!(*a.inner.borrow().grad.as_ref().unwrap().first().unwrap(), 2.0);
        // dy/db = a = 3.0
        assert_eq!(*b.inner.borrow().grad.as_ref().unwrap().first().unwrap(), 3.0);
    }

    /// 연산자 오버로딩: + 와 *
    #[test]
    fn test_operator_overload() {
        let a = Variable::new(arr0(3.0).into_dyn());
        let b = Variable::new(arr0(2.0).into_dyn());

        // 기존: add(&a, &b), mul(&a, &b)
        // 이제: &a + &b, &a * &b
        let y = &a + &b;
        assert_eq!(*y.inner.borrow().data.first().unwrap(), 5.0);

        let y = &a * &b;
        assert_eq!(*y.inner.borrow().data.first().unwrap(), 6.0);
    }

    /// 연산자 조합: y = a * b + c
    #[test]
    fn test_operator_combined() {
        let a = Variable::new(arr0(3.0).into_dyn());
        let b = Variable::new(arr0(2.0).into_dyn());
        let c = Variable::new(arr0(1.0).into_dyn());

        // y = a * b + c = 3 * 2 + 1 = 7
        let y = &(&a * &b) + &c;
        y.backward(false);

        assert_eq!(*y.inner.borrow().data.first().unwrap(), 7.0);
        // dy/da = b = 2
        assert_eq!(*a.inner.borrow().grad.as_ref().unwrap().first().unwrap(), 2.0);
        // dy/db = a = 3
        assert_eq!(*b.inner.borrow().grad.as_ref().unwrap().first().unwrap(), 3.0);
        // dy/dc = 1
        assert_eq!(*c.inner.borrow().grad.as_ref().unwrap().first().unwrap(), 1.0);
    }
}
