// dezero 라이브러리
// Python DeZero 프레임워크의 Rust 구현
// step23에서 패키지로 정리: 각 step 파일에서 중복되던 코드를 라이브러리로 추출

use ndarray::ArrayD;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt;
use std::rc::{Rc, Weak};

thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
}

// --- no_grad 모드 ---

pub struct NoGradGuard {
    prev: bool,
}

/// 역전파 그래프 생성을 비활성화하는 RAII 가드
/// let _guard = no_grad(); 형태로 사용, 스코프 종료 시 자동 복원
pub fn no_grad() -> NoGradGuard {
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
pub struct Variable {
    inner: Rc<RefCell<VarInner>>,
}

impl Variable {
    pub fn new(data: ArrayD<f64>) -> Self {
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

    pub fn with_name(data: ArrayD<f64>, name: &str) -> Self {
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

    pub fn set_name(&self, name: &str) {
        self.inner.borrow_mut().name = Some(name.to_string());
    }

    // --- 데이터 접근 ---

    pub fn data(&self) -> ArrayD<f64> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<ArrayD<f64>> {
        self.inner.borrow().grad.clone()
    }

    pub fn set_data(&self, data: ArrayD<f64>) {
        self.inner.borrow_mut().data = data;
    }

    // --- 형상 정보 ---

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.inner.borrow().data.ndim()
    }

    pub fn size(&self) -> usize {
        self.inner.borrow().data.len()
    }

    pub fn len(&self) -> usize {
        self.inner.borrow().data.shape().first().copied().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 거듭제곱 (Rust에는 ** 연산자가 없으므로 메서드로 제공)
    pub fn pow(&self, c: f64) -> Variable {
        powfn(self, c)
    }

    fn set_creator(&self, state: &FuncStateRef) {
        let func_gen = state.borrow().generation;
        let mut inner = self.inner.borrow_mut();
        inner.creator = Some(Rc::clone(state));
        inner.generation = func_gen + 1;
    }

    pub fn cleargrad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    pub fn backward(&self, retain_grad: bool) {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(ArrayD::ones(inner.data.shape()));
            }
        }

        let mut funcs: Vec<FuncStateRef> = Vec::new();
        let mut seen: HashSet<*const RefCell<FuncState>> = HashSet::new();

        let add_func = |f: FuncStateRef,
                        funcs: &mut Vec<FuncStateRef>,
                        seen: &mut HashSet<*const RefCell<FuncState>>| {
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

impl std::ops::Neg for &Variable {
    type Output = Variable;
    fn neg(self) -> Variable {
        neg(self)
    }
}

impl std::ops::Add for &Variable {
    type Output = Variable;
    fn add(self, rhs: Self) -> Variable {
        add(self, rhs)
    }
}

impl std::ops::Sub for &Variable {
    type Output = Variable;
    fn sub(self, rhs: Self) -> Variable {
        sub(self, rhs)
    }
}

impl std::ops::Mul for &Variable {
    type Output = Variable;
    fn mul(self, rhs: Self) -> Variable {
        mul(self, rhs)
    }
}

impl std::ops::Div for &Variable {
    type Output = Variable;
    fn div(self, rhs: Self) -> Variable {
        div(self, rhs)
    }
}

// 스칼라(f64)와의 연산

impl std::ops::Add<f64> for &Variable {
    type Output = Variable;
    fn add(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        add(self, &rhs)
    }
}

impl std::ops::Add<&Variable> for f64 {
    type Output = Variable;
    fn add(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        add(&lhs, rhs)
    }
}

impl std::ops::Sub<f64> for &Variable {
    type Output = Variable;
    fn sub(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        sub(self, &rhs)
    }
}

impl std::ops::Sub<&Variable> for f64 {
    type Output = Variable;
    fn sub(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        sub(&lhs, rhs)
    }
}

impl std::ops::Mul<f64> for &Variable {
    type Output = Variable;
    fn mul(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        mul(self, &rhs)
    }
}

impl std::ops::Mul<&Variable> for f64 {
    type Output = Variable;
    fn mul(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        mul(&lhs, rhs)
    }
}

impl std::ops::Div<f64> for &Variable {
    type Output = Variable;
    fn div(self, rhs: f64) -> Variable {
        let rhs = Variable::new(ndarray::arr0(rhs).into_dyn());
        div(self, &rhs)
    }
}

impl std::ops::Div<&Variable> for f64 {
    type Output = Variable;
    fn div(self, rhs: &Variable) -> Variable {
        let lhs = Variable::new(ndarray::arr0(self).into_dyn());
        div(&lhs, rhs)
    }
}

// --- Function 트레잇과 Func ---

pub trait Function {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    /// Python의 type(f).__class__.__name__ 에 해당
    fn name(&self) -> &str {
        "Function"
    }
}

pub struct Func {
    state: FuncStateRef,
}

impl Func {
    pub fn new(func: impl Function + 'static) -> Self {
        Func {
            state: Rc::new(RefCell::new(FuncState {
                func: Box::new(func),
                generation: 0,
                inputs: Vec::new(),
                outputs: Vec::new(),
            })),
        }
    }

    pub fn call(&self, inputs: &[&Variable]) -> Variable {
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

// --- 내장 함수 구현 ---

struct NegFn;

impl Function for NegFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![-&xs[0]]
    }
    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![-&gys[0]]
    }
    fn name(&self) -> &str { "Neg" }
}

struct AddFn;

impl Function for AddFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }
    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), gys[0].clone()]
    }
    fn name(&self) -> &str { "Add" }
}

struct SubFn;

impl Function for SubFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] - &xs[1]]
    }
    fn backward(&self, _xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![gys[0].clone(), -&gys[0]]
    }
    fn name(&self) -> &str { "Sub" }
}

struct MulFn;

impl Function for MulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[1]]
    }
    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[1] * &gys[0], &xs[0] * &gys[0]]
    }
    fn name(&self) -> &str { "Mul" }
}

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
    fn name(&self) -> &str { "Div" }
}

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
    fn name(&self) -> &str { "Pow" }
}

struct SinFn;

impl Function for SinFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::sin)]
    }
    fn backward(&self, xs: &[ArrayD<f64>], gys: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0].mapv(f64::cos) * &gys[0]]
    }
    fn name(&self) -> &str { "Sin" }
}

// --- 공개 함수 ---

pub fn neg(x: &Variable) -> Variable {
    Func::new(NegFn).call(&[x])
}

pub fn add(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(AddFn).call(&[x0, x1])
}

pub fn sub(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(SubFn).call(&[x0, x1])
}

pub fn mul(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(MulFn).call(&[x0, x1])
}

pub fn div(x0: &Variable, x1: &Variable) -> Variable {
    Func::new(DivFn).call(&[x0, x1])
}

pub fn powfn(x: &Variable, c: f64) -> Variable {
    Func::new(PowFn { c }).call(&[x])
}

pub fn sin(x: &Variable) -> Variable {
    Func::new(SinFn).call(&[x])
}

// --- 계산 그래프 시각화 (DOT/Graphviz) ---

/// Variable 노드의 DOT 표현
fn dot_var(v: &Variable, verbose: bool) -> String {
    let inner = v.inner.borrow();
    let id = Rc::as_ptr(&v.inner) as usize;
    let mut label = inner.name.clone().unwrap_or_default();
    if verbose {
        if inner.name.is_some() {
            label.push_str(": ");
        }
        label.push_str(&format!("{:?} f64", inner.data.shape()));
    }
    format!(
        "{} [label=\"{}\", color=orange, style=filled]\n",
        id, label
    )
}

/// Function 노드의 DOT 표현 (노드 + 입출력 엣지)
fn dot_func(state: &FuncState, state_ptr: usize) -> String {
    let mut txt = format!(
        "{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n",
        state_ptr,
        state.func.name()
    );
    for input in &state.inputs {
        let input_id = Rc::as_ptr(&input.inner) as usize;
        txt.push_str(&format!("{} -> {}\n", input_id, state_ptr));
    }
    for output in &state.outputs {
        if let Some(out) = output.upgrade() {
            let output_id = Rc::as_ptr(&out) as usize;
            txt.push_str(&format!("{} -> {}\n", state_ptr, output_id));
        }
    }
    txt
}

/// 계산 그래프를 DOT 형식 문자열로 변환
/// Python의 get_dot_graph에 해당
pub fn get_dot_graph(output: &Variable, verbose: bool) -> String {
    let mut txt = String::new();
    let mut funcs: Vec<FuncStateRef> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();

    let add_func = |f: FuncStateRef, funcs: &mut Vec<FuncStateRef>, seen: &mut HashSet<usize>| {
        let ptr = Rc::as_ptr(&f) as usize;
        if !seen.contains(&ptr) {
            seen.insert(ptr);
            funcs.push(f);
            funcs.sort_by_key(|f| f.borrow().generation);
        }
    };

    txt.push_str(&dot_var(output, verbose));

    if let Some(creator) = output.inner.borrow().creator.clone() {
        add_func(creator, &mut funcs, &mut seen);
    }

    while let Some(state_ref) = funcs.pop() {
        let state = state_ref.borrow();
        let state_ptr = Rc::as_ptr(&state_ref) as usize;
        txt.push_str(&dot_func(&state, state_ptr));

        for input in &state.inputs {
            txt.push_str(&dot_var(input, verbose));
            if let Some(creator) = input.inner.borrow().creator.clone() {
                add_func(creator, &mut funcs, &mut seen);
            }
        }
    }

    format!("digraph g {{\n{}}}", txt)
}

/// DOT 그래프를 파일로 저장하고 Graphviz로 이미지 생성
/// Python의 plot_dot_graph에 해당
pub fn plot_dot_graph(output: &Variable, verbose: bool, to_file: &str) -> std::io::Result<()> {
    let dot = get_dot_graph(output, verbose);

    // .dot 파일 경로 생성
    let dot_file = if let Some(stem) = to_file.strip_suffix(".png") {
        format!("{}.dot", stem)
    } else if let Some(stem) = to_file.strip_suffix(".pdf") {
        format!("{}.dot", stem)
    } else {
        format!("{}.dot", to_file)
    };

    std::fs::write(&dot_file, &dot)?;

    // 출력 형식 결정
    let ext = std::path::Path::new(to_file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png");

    // dot 명령 실행
    std::process::Command::new("dot")
        .args([&format!("-T{}", ext), &dot_file, "-o", to_file])
        .status()?;

    Ok(())
}
