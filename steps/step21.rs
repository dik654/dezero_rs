// step21: 연산자 오버로딩 확장
// step20의 +, * 에 더해 -(부정), -(뺄셈), /(나눗셈), pow(거듭제곱) 추가
// 스칼라(f64)와의 연산도 지원: &x + 2.0, 3.0 * &x 등

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

    /// 거듭제곱 (Rust에는 ** 연산자가 없으므로 메서드로 제공)
    fn pow(&self, c: f64) -> Variable {
        powfn(self, c)
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

// Variable 간 연산

/// -&x (부정)
impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        neg(self)
    }
}

/// &x + &y
impl std::ops::Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: Self) -> Variable {
        add(self, rhs)
    }
}

/// &x - &y
impl std::ops::Sub for &Variable {
    type Output = Variable;
    fn sub(self, rhs: Self) -> Variable {
        sub(self, rhs)
    }
}

/// &x * &y
impl std::ops::Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: Self) -> Variable {
        mul(self, rhs)
    }
}

/// &x / &y
impl std::ops::Div for &Variable {
    type Output = Variable;
    fn div(self, rhs: Self) -> Variable {
        div(self, rhs)
    }
}

// 스칼라(f64)와의 연산
// f64를 Variable로 감싸서 기존 함수를 재활용
// 좌우 순서를 둘 다 지원
/// &x + 2.0
impl std::ops::Add<f64> for &Variable {
    type Output = Variable;
    fn add(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        add(self, &rhs)
    }
}

/// 2.0 + &x
impl std::ops::Add<&Variable> for f64 {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        add(&lhs, rhs)
    }
}

/// &x - 2.0
impl std::ops::Sub<f64> for &Variable {
    type Output = Variable;
    fn sub(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        sub(self, &rhs)
    }
}

/// 2.0 - &x
impl std::ops::Sub<&Variable> for f64 {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        sub(&lhs, rhs)
    }
}

/// &x * 2.0
impl std::ops::Mul<f64> for &Variable {
    type Output = Variable;
    fn mul(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        mul(self, &rhs)
    }
}

/// 2.0 * &x
impl std::ops::Mul<&Variable> for f64 {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        mul(&lhs, rhs)
    }
}

/// &x / 2.0
impl std::ops::Div<f64> for &Variable {
    type Output = Variable;
    fn div(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        div(self, &rhs)
    }
}

/// 2.0 / &x
impl std::ops::Div<&Variable> for f64 {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        div(&lhs, rhs)
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

/// 부정: y = -x
/// backward: dy/dx = -1
struct NegFn;

impl Function for NegFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![-&xs[0]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![-&gys[0]]
    }
}

/// 덧셈: y = x0 + x1
struct AddFn;

impl Function for AddFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
}

/// 뺄셈: y = x0 - x1
/// backward: dy/dx0 = 1, dy/dx1 = -1
struct SubFn;

impl Function for SubFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] - &xs[1]]
    }

    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), -&gys[0]]
    }
}

/// 곱셈: y = x0 * x1
/// backward: dy/dx0 = x1, dy/dx1 = x0
struct MulFn;

impl Function for MulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[1]]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[1] * &gys[0], &xs[0] * &gys[0]]
    }
}

/// 나눗셈: y = x0 / x1
/// backward: dy/dx0 = 1/x1, dy/dx1 = -x0/x1²
struct DivFn;

impl Function for DivFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] / &xs[1]]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let gx0 = &gys[0] / &xs[1];
        let gx1 = -&gys[0] * &xs[0] / (&xs[1] * &xs[1]);
        vec![gx0, gx1]
    }
}

/// 거듭제곱: y = x^c (c는 상수)
/// backward: dy/dx = c * x^(c-1)
struct PowFn {
    c: f64,
}

impl Function for PowFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(|x| x.powf(self.c))]
    }

    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let c = self.c;
        vec![c * &xs[0].mapv(|x| x.powf(c - 1.0)) * &gys[0]]
    }
}

fn square(x: &Variable) -> Variable {
    Func::new(Square).call(&[x])
}

fn neg(x: &Variable) -> Variable {
    Func::new(NegFn).call(&[x])
}

fn add(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(AddFn).call(&[x0, x1])
}

fn sub(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(SubFn).call(&[x0, x1])
}

fn mul(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(MulFn).call(&[x0, x1])
}

fn div(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(DivFn).call(&[x0, x1])
}

fn powfn(x: &Variable, c: f64) -> Variable {
    Func::new(PowFn { c }).call(&[x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    fn get_val(v: &Variable) -> f64 {
        *v.inner.borrow().data.first().unwrap()
    }

    fn get_grad(v: &Variable) -> f64 {
        *v.inner.borrow().grad.as_ref().unwrap().first().unwrap()
    }

    /// 부정: y = -x
    #[test]
    fn test_neg() {
        let x = Variable::new(arr0(3.0).into_dyn());
        let y = -&x;
        y.backward(false);
        assert_eq!(get_val(&y), -3.0);
        assert_eq!(get_grad(&x), -1.0);
    }

    /// 뺄셈: y = a - b
    #[test]
    fn test_sub() {
        let a = Variable::new(arr0(5.0).into_dyn());
        let b = Variable::new(arr0(3.0).into_dyn());
        let y = &a - &b;
        y.backward(false);
        assert_eq!(get_val(&y), 2.0);
        assert_eq!(get_grad(&a), 1.0);
        assert_eq!(get_grad(&b), -1.0);
    }

    /// 나눗셈: y = a / b
    #[test]
    fn test_div() {
        let a = Variable::new(arr0(6.0).into_dyn());
        let b = Variable::new(arr0(3.0).into_dyn());
        let y = &a / &b;
        y.backward(false);
        assert_eq!(get_val(&y), 2.0);
        // dy/da = 1/b = 1/3
        assert!((get_grad(&a) - 1.0 / 3.0).abs() < 1e-10);
        // dy/db = -a/b² = -6/9 = -2/3
        assert!((get_grad(&b) - (-2.0 / 3.0)).abs() < 1e-10);
    }

    /// 거듭제곱: y = x^3
    #[test]
    fn test_pow() {
        let x = Variable::new(arr0(2.0).into_dyn());
        let y = x.pow(3.0);
        y.backward(false);
        assert_eq!(get_val(&y), 8.0);
        // dy/dx = 3x² = 12
        assert_eq!(get_grad(&x), 12.0);
    }

    /// 스칼라 연산: &x + 2.0, 3.0 * &x 등
    #[test]
    fn test_scalar_ops() {
        let x = Variable::new(arr0(3.0).into_dyn());

        assert_eq!(get_val(&(&x + 1.0)), 4.0);
        assert_eq!(get_val(&(1.0 + &x)), 4.0);
        assert_eq!(get_val(&(&x - 1.0)), 2.0);
        assert_eq!(get_val(&(10.0 - &x)), 7.0);
        assert_eq!(get_val(&(&x * 2.0)), 6.0);
        assert_eq!(get_val(&(2.0 * &x)), 6.0);
        assert_eq!(get_val(&(&x / 2.0)), 1.5);
        assert_eq!(get_val(&(6.0 / &x)), 2.0);
    }

    /// 복합 수식: y = x^3 - 2*x + 1 의 역전파
    #[test]
    fn test_complex_expression() {
        let x = Variable::new(arr0(2.0).into_dyn());
        // y = x³ - 2x + 1 = 8 - 4 + 1 = 5
        let y = &(&x.pow(3.0) - &(&x * 2.0)) + &Variable::new(arr0(1.0).into_dyn());
        y.backward(false);
        assert_eq!(get_val(&y), 5.0);
        // dy/dx = 3x² - 2 = 12 - 2 = 10
        assert_eq!(get_grad(&x), 10.0);
    }
}
