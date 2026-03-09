// dezero 라이브러리
// Python DeZero 프레임워크의 Rust 구현
// step23에서 패키지로 정리: 각 step 파일에서 중복되던 코드를 라이브러리로 추출

use ndarray::ArrayD;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt;
use std::io::Read;
use std::rc::{Rc, Weak};

thread_local! {
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };
    static TRAINING: Cell<bool> = const { Cell::new(true) };
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

/// Python의 using_config('enable_backprop', value)에 해당
/// 역전파 그래프 생성을 enable 값으로 설정하고, 스코프 종료 시 이전 값 복원
fn using_backprop(enable: bool) -> NoGradGuard {
    let prev = ENABLE_BACKPROP.with(|c| c.get());
    ENABLE_BACKPROP.with(|c| c.set(enable));
    NoGradGuard { prev }
}

// --- test_mode (훈련/추론 모드 전환) ---

pub struct TestModeGuard {
    prev: bool,
}

/// 추론(테스트) 모드로 전환하는 RAII 가드
/// Python의 with test_mode():에 해당
/// 훈련 시 동작이 달라지는 연산(dropout 등)을 제어
/// let _guard = test_mode(); 형태로 사용, 스코프 종료 시 자동 복원
pub fn test_mode() -> TestModeGuard {
    let prev = TRAINING.with(|c| c.get());
    TRAINING.with(|c| c.set(false));
    TestModeGuard { prev }
}

impl Drop for TestModeGuard {
    fn drop(&mut self) {
        TRAINING.with(|c| c.set(self.prev));
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
    // ArrayD가 아닌 Variable로 저장하는 이유:
    // 기울기도 cos, mul 같은 연산을 거쳐 만들어진 값
    // 그 연산 이력(creator 체인)을 보존해야 다시 미분할 수 있다
    //
    // 예) f(x) = x^4 - 2x^2, x = 2.0
    //
    //   ArrayD일 때는 단순히 숫자 값만 가지고 있어 문맥 정보가 없음:  x.grad = 24.0 
    //   Variable일 때: x.grad = Variable {
    //                    data: 24.0,
    //                    creator: SubFn ← MulFn ← PowFn ← x
    //                  }
    //                  → "24.0은 4x³-4x를 계산해서 나온 값"이라는 정보가 남아있음
    //                  → grad.backward() 하면 이 체인을 따라 f''(x) = 12x²-4 자동 계산
    grad: Option<Variable>,
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

    /// 기울기를 ArrayD로 반환 (하위 호환성)
    pub fn grad(&self) -> Option<ArrayD<f64>> {
        // Variable에서 data만 추출
        self.inner.borrow().grad.as_ref().map(|g| g.data())
    }

    /// 기울기를 Variable로 반환 (이중 역전파용)
    pub fn grad_var(&self) -> Option<Variable> {
        // 같은 Variable 데이터를 가리키는 포인터
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

    pub fn backward(&self, retain_grad: bool, create_graph: bool) {
        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(Variable::new(ArrayD::ones(inner.data.shape())));
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
            {
                // using_config('enable_backprop', create_graph)
                // create_graph=true: 역전파 계산도 그래프에 기록 (이중 역전파 가능)
                // create_graph=false: 역전파 계산 시 그래프 생성 비활성화
                let _guard = using_backprop(create_graph);

                let (gxs, inputs) = {
                    let state = state_ref.borrow();
                    let gys: Vec<Variable> = state
                        .outputs
                        .iter()
                        .map(|o| o.upgrade().unwrap().borrow().grad.clone().unwrap())
                        .collect();
                    let xs: Vec<Variable> = state.inputs.clone();
                    let inputs = state.inputs.clone();
                    let gxs = state.func.backward(&xs, &gys);
                    (gxs, inputs)
                };

                for (input, gx) in inputs.iter().zip(gxs) {
                    let mut inner = input.inner.borrow_mut();
                    if inner.grad.is_none() {
                        inner.grad = Some(gx);
                    } else {
                        let prev = inner.grad.take().unwrap();
                        inner.grad = Some(&prev + &gx);
                    }
                    drop(inner);
                    if let Some(creator) = input.inner.borrow().creator.clone() {
                        add_func(creator, &mut funcs, &mut seen);
                    }
                }
            }

            if !retain_grad {
                let state = state_ref.borrow();
                for output in &state.outputs {
                    if let Some(out) = output.upgrade() {
                        out.borrow_mut().grad = None;
                    }
                }
            }
        }
    }

    /// 계산 그래프를 절단하여 역전파 전파를 막음 (Truncated BPTT용)
    ///
    /// RNN 학습에서 시간 스텝이 길어지면 역전파 비용이 무한히 증가.
    /// loss.backward() 후 loss.unchain_backward()를 호출하면:
    ///   - 현재 loss에서 거슬러 올라가며 모든 중간 Variable의 creator를 제거
    ///   - 다음 backward에서는 절단된 지점 이전으로 역전파되지 않음
    ///   - 은닉 상태 h는 값은 유지되지만 그래프 연결이 끊김 → 상수 취급
    pub fn unchain_backward(&self) {
        if let Some(creator) = self.inner.borrow().creator.clone() {
            let mut funcs = vec![creator];
            while let Some(state_ref) = funcs.pop() {
                let state = state_ref.borrow();
                for input in &state.inputs {
                    let mut inner = input.inner.borrow_mut();
                    if let Some(c) = inner.creator.take() {
                        funcs.push(c);
                    }
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
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable>;
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
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![neg(&gys[0])]
    }
    fn name(&self) -> &str { "Neg" }
}

// 브로드캐스트 역전파: 기울기를 원래 shape로 축소(기울기는 원래 변수와 같은 shape여야 하기 때문)
// forward에서 shape (1,) + shape (3,) = shape (3,) 처럼 브로드캐스트되면
// backward에서 기울기 shape (3,)를 다시 shape (1,)로 줄여야 한다
//
// 예시) x=[1,1,1] shape(3,)를 target_shape (1,)로 축소하는 흐름:
//   x_ndim=1, t_ndim=1
//   padded_target = [] + [1] = [1]
//   axis=0: padded[0]=1, result.shape[0]=3 → 합산 → [3] shape(1,)
//   into_shape_with_order → [3] shape(1,)
fn sum_to(x: &Variable, target_shape: &[usize]) -> Variable {
    let x_shape = x.shape();
    if x_shape == target_shape {
        return x.clone();
    }
    let x_data = x.data();
    let x_ndim = x_data.ndim();
    let t_ndim = target_shape.len();

    // target_shape 앞에 1을 채워서 x와 차원 수를 맞춤
    // 예시) x_ndim=1, t_ndim=1 → 앞에 0개 채움 → padded = [1]
    let mut padded_target = vec![1usize; x_ndim.saturating_sub(t_ndim)];
    padded_target.extend_from_slice(target_shape);

    // 각 축을 비교해서, target이 1이고 x가 1이 아닌 축을 합산
    // 예시) axis=0: padded[0]=1, result.shape[0]=3 → 불일치 → 합산
    //   [1,1,1] → sum_axis(0) → 3 → insert_axis(0) → [3] shape(1,)
    let mut result = x_data.clone();
    for axis in (0..x_ndim).rev() {
        if padded_target[axis] == 1 && result.shape()[axis] != 1 {
            // sum_axis: 해당 축 방향으로 합산 (축 제거됨)
            // insert_axis: 제거된 축을 크기 1로 다시 삽입 (차원 수 유지)
            result = result.sum_axis(ndarray::Axis(axis)).insert_axis(ndarray::Axis(axis));
        }
    }
    // 최종 shape를 target_shape로 맞춤
    // 예시) [3] shape(1,) → into_shape_with_order((1,)) → [3] shape(1,)
    let result = result.into_shape_with_order(ndarray::IxDyn(target_shape)).unwrap();
    Variable::new(result)
}

struct AddFn;

impl Function for AddFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] + &xs[1]]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        // 첫 번째 입력 복구 index 0
        let gx0 = sum_to(&gys[0], &xs[0].shape());
        // 두 번째 입력 복구 index 1
        let gx1 = sum_to(&gys[0], &xs[1].shape());
        vec![gx0, gx1]
    }
    fn name(&self) -> &str { "Add" }
}

struct SubFn;

impl Function for SubFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] - &xs[1]]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gx0 = sum_to(&gys[0], &xs[0].shape());
        let gx1 = sum_to(&neg(&gys[0]), &xs[1].shape());
        vec![gx0, gx1]
    }
    fn name(&self) -> &str { "Sub" }
}

struct MulFn;

impl Function for MulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] * &xs[1]]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gx0 = sum_to(&(&xs[1] * &gys[0]), &xs[0].shape());
        let gx1 = sum_to(&(&xs[0] * &gys[0]), &xs[1].shape());
        vec![gx0, gx1]
    }
    fn name(&self) -> &str { "Mul" }
}

struct DivFn;

impl Function for DivFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![&xs[0] / &xs[1]]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gx0 = sum_to(&(&gys[0] / &xs[1]), &xs[0].shape());
        let gx1 = sum_to(&(&(&neg(&gys[0]) * &xs[0]) / &(&xs[1] * &xs[1])), &xs[1].shape());
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
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let c = self.c;
        vec![&(c * &xs[0].pow(c - 1.0)) * &gys[0]]
    }
    fn name(&self) -> &str { "Pow" }
}

struct SinFn;

impl Function for SinFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::sin)]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&cos(&xs[0]) * &gys[0]]
    }
    fn name(&self) -> &str { "Sin" }
}

struct CosFn;

impl Function for CosFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::cos)]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&neg(&sin(&xs[0])) * &gys[0]]
    }
    fn name(&self) -> &str { "Cos" }
}

struct TanhFn;

impl Function for TanhFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::tanh)]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let y: Variable = tanh(&xs[0]);
        // tanh(x)의 미분은 tanh'(x) = 1 - tanh(x)^2
        vec![&gys[0] * &(1.0 - &(&y * &y))]
    }
    fn name(&self) -> &str { "Tanh" }
}

/// 배열의 합산 연산 (axis, keepdims 지원)
/// axis=None: 전체 합산 → 스칼라        [1,2,3,4,5,6] -> 21
/// axis=Some(0): 특정 축 방향으로만 합산 → [[1,2,3],[4,5,6]] → [5,7,9]
/// keepdims=true: 합산 후에도 사라질 축을 제거하지 않고 크기 1로 남겨서 차원 수 유지 → shape (2,3,4,5) → (1,1,1,1)
struct SumFn {
    axis: Option<usize>,
    keepdims: bool,
    x_shape: Vec<usize>, // backward에서 기울기를 원래 shape로 복원하기 위해 저장
}

impl Function for SumFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        match self.axis {
            None => {
                // 전체 합산 → 스칼라
                let s = xs[0].sum();
                // keepdims: 입력과 같은 차원 수를 유지 (각 축 크기 1)
                if self.keepdims {
                    // 배열 차원수 가져오기
                    let ndim = xs[0].ndim();
                    // 차원 shape 구성
                    // ndim 차원에 축의 크기가 모두 1
                    let shape = vec![1; ndim];
                    // shape와 1차원 Vec을 받아서 배열을 만들기
                    // ndarray::ArrayD::from_shape_vec(
                    //     ndarray::IxDyn(&[2, 3]),           // shape: 2행 3열
                    //     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] // data: 실제 값
                    // )
                    // → [[1, 2, 3],
                    //    [4, 5, 6]]
                    vec![ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), vec![s]).unwrap()]
                } else {
                    // 단일 스칼라 값으로 
                    vec![ndarray::arr0(s).into_dyn()]
                }
            }
            Some(axis) => {
                // 특정 축 방향으로 합산
                let summed = xs[0].sum_axis(ndarray::Axis(axis));
                if self.keepdims {
                    // 합산으로 사라진 축을 크기 1로 다시 삽입
                    let mut shape: Vec<usize> = summed.shape().to_vec();
                    shape.insert(axis, 1);
                    vec![summed.into_shape_with_order(ndarray::IxDyn(&shape)).unwrap()]
                } else {
                    vec![summed]
                }
            }
        }
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = &gys[0];
        let gy_data = gy.data();

        // keepdims=false로 축이 제거된 경우, broadcast 전에 축을 다시 삽입
        // 예: (B,T,D) → sum(axis=1) → (B,D) → backward → (B,1,D) → broadcast → (B,T,D)
        let gy_for_broadcast = if let Some(axis) = self.axis {
            if !self.keepdims {
                let mut shape = gy_data.shape().to_vec();
                shape.insert(axis, 1);
                gy_data.to_owned().into_shape_with_order(ndarray::IxDyn(&shape)).unwrap()
            } else {
                gy_data.to_owned()
            }
        } else {
            gy_data.to_owned()
        };

        let broadcast = gy_for_broadcast.broadcast(ndarray::IxDyn(&self.x_shape)).unwrap().to_owned();
        vec![Variable::new(broadcast)]
    }
    fn name(&self) -> &str { "Sum" }
}

// reshape는 값을 바꾸지 않고 shape만 바꾸는 연산
// 순전파: x (2,3) -> y (6,)   데이터 [0,1,2,3,4,5]는 동일
// 역전파: 기울기도 값은 그대로, shape만 되돌리면 됨
//   gys[0] shape (6,) -> gx shape (2,3)
//   dy/dx_ij = 1 (각 원소가 그대로 출력에 매핑되므로)
struct ReshapeFn {
    target_shape: Vec<usize>,
    // reshape() 호출 시점에 입력의 원래 shape를 캡처해서 여기에 저장한다.
    // backward에서 이 값으로 기울기를 원래 shape로 되돌린다.
    x_shape: Vec<usize>,
}

impl Function for ReshapeFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let reshaped = xs[0].clone()
            .into_shape_with_order(ndarray::IxDyn(&self.target_shape))
            .unwrap();
        vec![reshaped.to_owned()]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        // 저장해둔 x_shape로 기울기를 원래 shape로 되돌린다
        vec![reshape(&gys[0], &self.x_shape)]
    }
    fn name(&self) -> &str { "Reshape" }
}

// transpose는 행과 열을 뒤집는 연산
// transpose(transpose(x)) = x 이므로 원래 shape를 저장할 필요 없이
// backward에서 다시 transpose하면 원래 shape로 돌아간다.
//
// 순전파: x (2,3) -> y (3,2)
//   x = [[1,2,3],       y = [[1,4],
//        [4,5,6]]            [2,5],
//                             [3,6]]
//
// 역전파: gys[0] (3,2) -> transpose -> gx (2,3)
//   x[0][1]=2 가 y[1][0]=2 로 갔으므로
//   y[1][0]의 기울기는 x[0][1]의 기울기 -> 다시 transpose하면 원래 위치
struct TransposeFn;

impl Function for TransposeFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].t().to_owned().into_dyn()]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![transpose(&gys[0])]
    }
    fn name(&self) -> &str { "Transpose" }
}

// 행렬곱: y = x @ w
// 순전파: x (2,3) @ w (3,4) = y (2,4)
// 역전파: gx = gy @ w^T   (2,4) @ (4,3) = (2,3)
//         gw = x^T @ gy    (3,2) @ (2,4) = (3,4)
struct MatMulFn;

impl Function for MatMulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = xs[0].view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w = xs[1].view().into_dimensionality::<ndarray::Ix2>().unwrap();
        vec![x.dot(&w).into_dyn()]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gx = matmul(&gys[0], &transpose(&xs[1])); // gy @ w^T
        let gw = matmul(&transpose(&xs[0]), &gys[0]);  // x^T @ gy
        vec![gx, gw]
    }
    fn name(&self) -> &str { "MatMul" }
}

struct SigmoidFn;

impl Function for SigmoidFn {
    // sigmoid: σ(x) = 1 / (1 + exp(-x))
    // 순전파: 각 원소에 시그모이드 함수 적용 → 출력이 (0, 1) 범위
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(|x| 1.0 / (1.0 + (-x).exp()))]
    }
    // 역전파: σ'(x) = σ(x) * (1 - σ(x))
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let y = sigmoid(&xs[0]);
        vec![&gys[0] * &(&y * &(1.0 - &y))]
    }
    fn name(&self) -> &str { "Sigmoid" }
}

struct ExpFn;

impl Function for ExpFn {
    // exp: y = e^x
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::exp)]
    }
    // 역전파: exp'(x) = exp(x)  (미분해도 자기 자신)
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&gys[0] * &exp(&xs[0])]
    }
    fn name(&self) -> &str { "Exp" }
}

struct LogFn;

impl Function for LogFn {
    // log: y = ln(x)
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::ln)]
    }
    // 역전파: log'(x) = 1/x
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&gys[0] / &xs[0]]
    }
    fn name(&self) -> &str { "Log" }
}

/// Softmax Cross-Entropy: 분류 문제의 손실 함수
/// softmax로 확률 변환 후 정답 클래스의 -log(확률)의 평균
/// softmax와 cross-entropy를 하나로 합쳐서 수치적으로 안정
struct SoftmaxCrossEntropyFn {
    t: Vec<usize>, // 정답 클래스 인덱스 (각 샘플마다 하나)
}

impl Function for SoftmaxCrossEntropyFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0]; // (N, C)
        let n = x.shape()[0];
        let c = x.shape()[1];

        // 수치 안정성: 각 행에서 최댓값을 빼서 exp overflow 방지
        // exp(100)은 overflow지만 exp(100-100) = exp(0) = 1
        let mut softmax = ArrayD::zeros(x.raw_dim());
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c {
                max_val = max_val.max(x[[i, j]]);
            }
            let mut sum_exp = 0.0;
            for j in 0..c {
                let e = (x[[i, j]] - max_val).exp();
                softmax[[i, j]] = e;
                sum_exp += e;
            }
            for j in 0..c {
                softmax[[i, j]] /= sum_exp;
            }
        }

        // cross-entropy: -mean(log(p[i, t[i]]))
        let mut loss = 0.0;
        for i in 0..n {
            let p = softmax[[i, self.t[i]]].max(1e-15); // log(0) 방지
            loss -= p.ln();
        }
        loss /= n as f64;

        vec![ndarray::arr0(loss).into_dyn()]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let x_data = xs[0].data(); // (N, C)
        let n = x_data.shape()[0];
        let c = x_data.shape()[1];

        // softmax 재계산 (수치 안정 버전)
        let mut softmax = ArrayD::zeros(x_data.raw_dim());
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c {
                max_val = max_val.max(x_data[[i, j]]);
            }
            let mut sum_exp = 0.0;
            for j in 0..c {
                let e = (x_data[[i, j]] - max_val).exp();
                softmax[[i, j]] = e;
                sum_exp += e;
            }
            for j in 0..c {
                softmax[[i, j]] /= sum_exp;
            }
        }

        // gradient: (softmax - one_hot) / N
        // 정답 클래스 위치에서만 1을 빼는 것이 one_hot 역할
        for i in 0..n {
            softmax[[i, self.t[i]]] -= 1.0;
        }
        let gx = softmax.mapv(|v| v / n as f64);

        // upstream gradient 곱하기 (scalar)
        let gy_val = gys[0].data().iter().next().copied().unwrap_or(1.0);
        let gx = gx.mapv(|v| v * gy_val);

        vec![Variable::new(gx)]
    }

    fn name(&self) -> &str { "SoftmaxCrossEntropy" }
}

/// Dropout: 훈련 시 뉴런을 무작위로 비활성화하는 정규화 기법
/// Inverted Dropout 방식:
///   훈련: y = x * mask / (1 - p)  (mask: 확률 p로 0, 아니면 1)
///   추론: y = x  (그대로 통과)
/// mask를 RefCell에 저장하여 backward에서 재사용
struct DropoutFn {
    dropout_ratio: f64,
    mask: RefCell<ArrayD<f64>>,
}

impl Function for DropoutFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        // mask 생성: 각 원소가 dropout_ratio보다 크면 1, 아니면 0
        let scale = 1.0 / (1.0 - self.dropout_ratio);
        let mask = DROPOUT_RNG.with(|rng| {
            let mut state = rng.get();
            ArrayD::from_shape_fn(x.raw_dim(), |_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = (state >> 11) as f64 / (1u64 << 53) as f64;
                rng.set(state);
                if r > self.dropout_ratio { scale } else { 0.0 }
            })
        });
        let y = x * &mask;
        *self.mask.borrow_mut() = mask;
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let mask = self.mask.borrow();
        vec![&gys[0] * &Variable::new(mask.clone())]
    }

    fn name(&self) -> &str { "Dropout" }
}

thread_local! {
    static DROPOUT_RNG: Cell<u64> = const { Cell::new(1234567890) };
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

pub fn cos(x: &Variable) -> Variable {
    Func::new(CosFn).call(&[x])
}

pub fn tanh(x: &Variable) -> Variable {
    Func::new(TanhFn).call(&[x])
}

/// 모든 원소를 더해 스칼라 하나로 만듦
pub fn sum(x: &Variable) -> Variable {
    let x_shape = x.shape();
    Func::new(SumFn { axis: None, keepdims: false, x_shape }).call(&[x])
}

/// 옵션 지정 가능
pub fn sum_with(x: &Variable, axis: Option<usize>, keepdims: bool) -> Variable {
    let x_shape = x.shape();
    Func::new(SumFn { axis, keepdims, x_shape }).call(&[x])
}

pub fn reshape(x: &Variable, shape: &[usize]) -> Variable {
    // 여기서 원래 shape를 캡처해서 ReshapeFn.x_shape에 저장
    let x_shape = x.shape();
    Func::new(ReshapeFn { target_shape: shape.to_vec(), x_shape }).call(&[x])
}

pub fn transpose(x: &Variable) -> Variable {
    Func::new(TransposeFn).call(&[x])
}

/// 임의 축 순열 전치: axes로 차원 순서를 재배치
/// 예: (B, H, T, D).transpose_axes([0, 2, 1, 3]) → (B, T, H, D)
///
/// 2D 전치(TransposeFn)는 (M, N) → (N, M)만 가능하지만,
/// Attention에서는 (B, H, T, D) → (B, T, H, D) 같은 임의 축 순열이 필수
///
/// 역전파: 역순열(inverse permutation)을 적용
///   axes = [0, 2, 1, 3]의 역순열도 [0, 2, 1, 3] (자기 자신의 역)
///   일반적으로 inv_axes[axes[i]] = i
struct TransposeAxesFn {
    axes: Vec<usize>,
    inv_axes: Vec<usize>,
}

impl Function for TransposeAxesFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let permuted = xs[0].clone().permuted_axes(ndarray::IxDyn(&self.axes));
        // permuted_axes 후 비연속 메모리일 수 있으므로 연속화
        vec![permuted.as_standard_layout().into_owned()]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![transpose_axes(&gys[0], &self.inv_axes)]
    }
    fn name(&self) -> &str { "TransposeAxes" }
}

pub fn transpose_axes(x: &Variable, axes: &[usize]) -> Variable {
    // 역순열 계산: inv_axes[axes[i]] = i
    let mut inv_axes = vec![0; axes.len()];
    for (i, &a) in axes.iter().enumerate() {
        inv_axes[a] = i;
    }
    Func::new(TransposeAxesFn {
        axes: axes.to_vec(),
        inv_axes,
    }).call(&[x])
}

/// Batched Matmul: 배치 차원을 유지한 채 마지막 2차원에서 행렬곱
/// (..., M, K) @ (..., K, N) → (..., M, N)
///
/// 일반 matmul(MatMulFn)은 2D (M,K)@(K,N) 전용.
/// Attention에서는 (B, H, T, D) @ (B, H, D, T) 같은 배치 행렬곱이 필수.
///
/// 역전파:
///   gy: (..., M, N)
///   gx = gy @ w^T   = batched_matmul(gy, w.transpose(-1,-2))
///   gw = x^T @ gy   = batched_matmul(x.transpose(-1,-2), gy)
struct BatchedMatMulFn;

impl Function for BatchedMatMulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        let w = &xs[1];
        let x_shape = x.shape();
        let w_shape = w.shape();
        let ndim = x_shape.len();

        let m = x_shape[ndim - 2];
        let k = x_shape[ndim - 1];
        let n = w_shape[ndim - 1];

        // 배치 크기 계산: 마지막 2차원을 제외한 모든 차원의 곱
        let batch: usize = x_shape[..ndim - 2].iter().product();

        // (..., M, K) → (batch, M, K) → 각 배치에서 (M,K) @ (K,N)
        // transpose_axes 후 비연속 메모리일 수 있으므로 as_standard_layout으로 연속화
        let x_contig = x.as_standard_layout().into_owned();
        let w_contig = w.as_standard_layout().into_owned();
        let x_flat = x_contig.view().into_shape_with_order(ndarray::IxDyn(&[batch, m, k])).unwrap();
        let w_flat = w_contig.view().into_shape_with_order(ndarray::IxDyn(&[batch, k, n])).unwrap();

        let mut out_data = Vec::with_capacity(batch * m * n);
        for b in 0..batch {
            let xb = x_flat.slice(ndarray::s![b, .., ..]);
            let wb = w_flat.slice(ndarray::s![b, .., ..]);
            let yb = xb.dot(&wb);
            out_data.extend(yb.iter());
        }

        let mut out_shape = x_shape[..ndim - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        vec![ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out_data).unwrap()]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        // gx = gy @ w^T
        let w_t = swap_last_two(&xs[1]);
        let gx = batched_matmul(&gys[0], &w_t);
        // gw = x^T @ gy
        let x_t = swap_last_two(&xs[0]);
        let gw = batched_matmul(&x_t, &gys[0]);
        vec![gx, gw]
    }
    fn name(&self) -> &str { "BatchedMatMul" }
}

/// 마지막 두 차원을 교환하는 헬퍼: (..., M, N) → (..., N, M)
fn swap_last_two(x: &Variable) -> Variable {
    let ndim = x.shape().len();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(ndim - 2, ndim - 1);
    transpose_axes(x, &axes)
}

pub fn batched_matmul(x: &Variable, w: &Variable) -> Variable {
    Func::new(BatchedMatMulFn).call(&[x, w])
}

/// Softmax(axis): 임의 축을 따라 softmax 적용
/// softmax(x)_i = exp(x_i - max) / Σ exp(x_j - max)
///
/// 기존 softmax_simple은 2D 텐서의 axis=1 전용.
/// Attention에서는 4D 텐서 (B, H, T, T)의 마지막 축(axis=-1)에 softmax 필요.
///
/// 수치 안정성: max를 빼서 exp의 오버플로 방지
/// 역전파: gx = y * (gy - sum(gy * y, axis, keepdims))
///   야코비안 ∂y_i/∂x_j = y_i(δ_ij - y_j)에서 유도
struct SoftmaxAxisFn {
    axis: usize,
    output: RefCell<ArrayD<f64>>,
}

impl Function for SoftmaxAxisFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        let axis = self.axis;

        // 수치 안정성: max를 빼서 가장 큰 값이 0이 되게 함
        let max_vals = x.map_axis(ndarray::Axis(axis), |lane| {
            lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });
        let mut max_shape = max_vals.shape().to_vec();
        max_shape.insert(axis, 1);
        let max_broadcast = max_vals
            .into_shape_with_order(ndarray::IxDyn(&max_shape))
            .unwrap();

        let shifted = x - &max_broadcast;
        let exp_x = shifted.mapv(f64::exp);

        let sum_exp = exp_x.sum_axis(ndarray::Axis(axis));
        let mut sum_shape = sum_exp.shape().to_vec();
        sum_shape.insert(axis, 1);
        let sum_broadcast = sum_exp
            .into_shape_with_order(ndarray::IxDyn(&sum_shape))
            .unwrap();

        let y = &exp_x / &sum_broadcast;
        *self.output.borrow_mut() = y.clone();
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        // gx = y * (gy - sum(gy * y, axis, keepdims=true))
        let y = self.output.borrow();
        let gy = gys[0].data();
        let axis = self.axis;

        let y_arr: &ArrayD<f64> = &y;
        let gy_y = &gy * y_arr;
        let sum_gy_y = gy_y.sum_axis(ndarray::Axis(axis));
        let mut sum_shape = sum_gy_y.shape().to_vec();
        sum_shape.insert(axis, 1);
        let sum_broadcast = sum_gy_y
            .into_shape_with_order(ndarray::IxDyn(&sum_shape))
            .unwrap();

        let diff = &gy - &sum_broadcast;
        let gx = y_arr * &diff;
        vec![Variable::new(gx)]
    }

    fn name(&self) -> &str { "Softmax" }
}

/// 임의 축에 대한 softmax
/// axis: 음수 지원 (-1 = 마지막 축)
pub fn softmax(x: &Variable, axis: isize) -> Variable {
    let ndim = x.shape().len();
    let axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
    Func::new(SoftmaxAxisFn {
        axis,
        output: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
    }).call(&[x])
}

/// Causal Mask: 미래 토큰을 차단하는 삼각 마스크
/// 마지막 두 차원 (T, T)에서 col > row인 위치를 -∞로 설정
///
/// Attention에서 scores (B, H, T, T)에 적용:
///   token i는 token 0..=i만 참조 가능 (자기자신 포함, 미래 불가)
///
/// 역전파: 마스크된 위치(-∞)의 기울기는 0, 나머지는 그대로 통과
struct CausalMaskFn;

impl Function for CausalMaskFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let mut y = xs[0].clone();
        let shape = y.shape().to_vec();
        let ndim = shape.len();
        let t_row = shape[ndim - 2];
        let t_col = shape[ndim - 1];

        // 배치 차원 평탄화: (..., T, T) → (batch, T, T)
        let batch: usize = shape[..ndim - 2].iter().product::<usize>().max(1);
        let stride = t_row * t_col;
        let y_slice = y.as_slice_mut().unwrap();
        for b in 0..batch {
            for i in 0..t_row {
                for j in (i + 1)..t_col {
                    y_slice[b * stride + i * t_col + j] = f64::NEG_INFINITY;
                }
            }
        }
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = gys[0].data();
        let shape = gy.shape().to_vec();
        let ndim = shape.len();
        let t_row = shape[ndim - 2];
        let t_col = shape[ndim - 1];

        let batch: usize = shape[..ndim - 2].iter().product::<usize>().max(1);
        let stride = t_row * t_col;
        let mut gx = gy.clone();
        let gx_slice = gx.as_slice_mut().unwrap();
        for b in 0..batch {
            for i in 0..t_row {
                for j in (i + 1)..t_col {
                    gx_slice[b * stride + i * t_col + j] = 0.0;
                }
            }
        }
        vec![Variable::new(gx)]
    }

    fn name(&self) -> &str { "CausalMask" }
}

/// Causal mask 적용: 미래 토큰 위치를 -∞로 마스킹
pub fn causal_mask(x: &Variable) -> Variable {
    Func::new(CausalMaskFn).call(&[x])
}

/// LayerNorm: 마지막 축(feature)을 따라 정규화
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// BatchNorm은 배치 방향으로 정규화하지만,
/// LayerNorm은 각 샘플 내에서 feature 방향으로 정규화.
/// 배치 크기에 의존하지 않아 Transformer에서 표준.
///
/// 역전파:
///   gbeta = sum(gy, batch_dims)
///   ggamma = sum(gy * x_hat, batch_dims)
///   gx = (1/sigma) * (g_xhat - mean(g_xhat) - x_hat * mean(g_xhat * x_hat))
///   where g_xhat = gy * gamma
struct LayerNormFn {
    eps: f64,
    x_hat: RefCell<ArrayD<f64>>,
    std_inv: RefCell<ArrayD<f64>>,
}

impl Function for LayerNormFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];      // (..., D)
        let gamma = &xs[1];  // (D,)
        let beta = &xs[2];   // (D,)
        let ndim = x.ndim();
        let last = ndim - 1;

        // mean along last axis → (...,) → reshape to (..., 1) for broadcast
        let mean = x.mean_axis(ndarray::Axis(last)).unwrap();
        let mut mean_shape = mean.shape().to_vec();
        mean_shape.push(1);
        let mean = mean
            .into_shape_with_order(ndarray::IxDyn(&mean_shape))
            .unwrap();

        let x_centered = x - &mean;

        // variance along last axis
        let var = x_centered.mapv(|v| v * v).mean_axis(ndarray::Axis(last)).unwrap();
        let mut var_shape = var.shape().to_vec();
        var_shape.push(1);
        let var = var
            .into_shape_with_order(ndarray::IxDyn(&var_shape))
            .unwrap();

        let std_inv = var.mapv(|v| 1.0 / (v + self.eps).sqrt());
        let x_hat = &x_centered * &std_inv;

        // y = gamma * x_hat + beta (gamma, beta는 (D,)이므로 마지막 축에 broadcast)
        let y = &x_hat * gamma + beta;

        *self.x_hat.borrow_mut() = x_hat;
        *self.std_inv.borrow_mut() = std_inv;
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = gys[0].data();          // (..., D)
        let gamma = _xs[1].data();       // (D,)
        let x_hat = self.x_hat.borrow(); // (..., D)
        let std_inv = self.std_inv.borrow(); // (..., 1)
        let ndim = gy.ndim();
        let d = gy.shape()[ndim - 1];

        // 배치 차원을 평탄화: (..., D) → (batch, D)
        let batch: usize = gy.shape()[..ndim - 1].iter().product::<usize>().max(1);

        // gbeta = sum(gy, batch_dims) → (D,)
        let gy_2d = gy
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(ndarray::IxDyn(&[batch, d]))
            .unwrap();
        let gbeta_arr = gy_2d.sum_axis(ndarray::Axis(0));

        // ggamma = sum(gy * x_hat, batch_dims) → (D,)
        let xh_2d = x_hat
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order(ndarray::IxDyn(&[batch, d]))
            .unwrap();
        let gy_xhat = &gy_2d * &xh_2d;
        let ggamma_arr = gy_xhat.sum_axis(ndarray::Axis(0));

        // gx: g_xhat = gy * gamma → (..., D)
        let g_xhat = &gy * &gamma;

        // mean(g_xhat, last_axis, keepdims) → (..., 1)
        let g_xhat_mean = g_xhat.mean_axis(ndarray::Axis(ndim - 1)).unwrap();
        let mut ms = g_xhat_mean.shape().to_vec();
        ms.push(1);
        let g_xhat_mean = g_xhat_mean
            .into_shape_with_order(ndarray::IxDyn(&ms))
            .unwrap();

        // mean(g_xhat * x_hat, last_axis, keepdims) → (..., 1)
        let x_hat_ref: &ArrayD<f64> = &x_hat;
        let g_xhat_xhat = &g_xhat * x_hat_ref;
        let g_xhat_xhat_mean = g_xhat_xhat.mean_axis(ndarray::Axis(ndim - 1)).unwrap();
        let mut ms2 = g_xhat_xhat_mean.shape().to_vec();
        ms2.push(1);
        let g_xhat_xhat_mean = g_xhat_xhat_mean
            .into_shape_with_order(ndarray::IxDyn(&ms2))
            .unwrap();

        // gx = std_inv * (g_xhat - mean(g_xhat) - x_hat * mean(g_xhat * x_hat))
        let term = &g_xhat - &g_xhat_mean - &(x_hat_ref * &g_xhat_xhat_mean);
        let std_inv_ref: &ArrayD<f64> = &std_inv;
        let gx = std_inv_ref * &term;

        vec![
            Variable::new(gx),
            Variable::new(ggamma_arr.into_dyn()),
            Variable::new(gbeta_arr.into_dyn()),
        ]
    }

    fn name(&self) -> &str { "LayerNorm" }
}

/// LayerNorm: 마지막 축에 대해 정규화
pub fn layer_norm(x: &Variable, gamma: &Variable, beta: &Variable, eps: f64) -> Variable {
    Func::new(LayerNormFn {
        eps,
        x_hat: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
        std_inv: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
    }).call(&[x, gamma, beta])
}

/// GELU: Gaussian Error Linear Unit
/// GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
///
/// ReLU의 "hard" 전환 대신 확률적으로 부드러운 전환.
/// x가 매우 음수면 ≈0, 매우 양수면 ≈x, 전환 구간에서 부드러운 S자.
/// GPT-2/3에서 사용하는 tanh 근사 버전.
///
/// 역전파:
///   u = √(2/π)(x + 0.044715x³)
///   du/dx = √(2/π)(1 + 0.134145x²)
///   gx = gy * [0.5(1 + tanh(u)) + 0.5x·sech²(u)·du/dx]
struct GELUFn;

impl Function for GELUFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
        let y = x.mapv(|v| {
            let u = sqrt_2_pi * (v + 0.044715 * v.powi(3));
            0.5 * v * (1.0 + u.tanh())
        });
        vec![y]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let x = xs[0].data();
        let gy = gys[0].data();
        let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();

        let gx = ndarray::Zip::from(&gy).and(&x).map_collect(|&gy_v, &x_v| {
            let u = sqrt_2_pi * (x_v + 0.044715 * x_v.powi(3));
            let tanh_u = u.tanh();
            let sech2_u = 1.0 - tanh_u * tanh_u;
            let du_dx = sqrt_2_pi * (1.0 + 0.134145 * x_v * x_v);
            gy_v * (0.5 * (1.0 + tanh_u) + 0.5 * x_v * sech2_u * du_dx)
        });

        vec![Variable::new(gx.into_dyn())]
    }

    fn name(&self) -> &str { "GELU" }
}

/// GELU 활성화: Transformer FFN의 표준 활성화 함수
pub fn gelu(x: &Variable) -> Variable {
    Func::new(GELUFn).call(&[x])
}

pub fn matmul(x: &Variable, w: &Variable) -> Variable {
    Func::new(MatMulFn).call(&[x, w])
}

pub fn sigmoid(x: &Variable) -> Variable {
    Func::new(SigmoidFn).call(&[x])
}

/// Dropout: 훈련 시 뉴런을 무작위 비활성화, 추론 시 통과
/// dropout_ratio: 비활성화 확률 (기본 0.5)
pub fn dropout(x: &Variable, dropout_ratio: f64) -> Variable {
    if TRAINING.with(|c| c.get()) {
        Func::new(DropoutFn {
            dropout_ratio,
            mask: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
        }).call(&[x])
    } else {
        x.clone()
    }
}

pub fn exp(x: &Variable) -> Variable {
    Func::new(ExpFn).call(&[x])
}

pub fn log(x: &Variable) -> Variable {
    Func::new(LogFn).call(&[x])
}

/// Softmax: 원시 점수를 확률로 변환 (각 행의 합 = 1)
/// axis=1 방향으로 정규화: exp(x) / sum(exp(x), axis=1)
/// 기존 연산(exp, sum_with, div)의 조합이므로 역전파는 자동 처리
pub fn softmax_simple(x: &Variable) -> Variable {
    let e = exp(x);
    let s = sum_with(&e, Some(1), true); // axis=1, keepdims=true → (N, 1)
    &e / &s // broadcast: (N, C) / (N, 1) → (N, C)
}

/// Softmax Cross-Entropy: 분류 문제의 표준 손실 함수
/// x: 모델의 원시 출력 (N, C), t: 정답 클래스 인덱스 [0, C)
/// 내부에서 softmax + cross-entropy를 한 번에 계산 (수치 안정)
pub fn softmax_cross_entropy_simple(x: &Variable, t: &[usize]) -> Variable {
    Func::new(SoftmaxCrossEntropyFn { t: t.to_vec() }).call(&[x])
}

/// 분류 정확도: 예측 클래스와 정답 클래스가 일치하는 비율
/// y: 모델 출력 (N, C) logits, t: 정답 클래스 인덱스
/// 각 행에서 argmax를 구해 정답과 비교
/// 평가 지표이므로 backward 불필요 → f64 반환 (Variable이 아님)
pub fn accuracy(y: &Variable, t: &[usize]) -> f64 {
    let y_data = y.data();
    let n = y_data.shape()[0];
    let c = y_data.shape()[1];

    let mut correct = 0;
    for i in 0..n {
        // argmax: 가장 큰 값의 인덱스 = 예측 클래스
        let mut max_j = 0;
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..c {
            if y_data[[i, j]] > max_val {
                max_val = y_data[[i, j]];
                max_j = j;
            }
        }
        if max_j == t[i] {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

/// 합성곱 출력 크기 계산
/// (input_size + 2*pad - kernel_size) / stride + 1
pub fn get_conv_outsize(input_size: usize, kernel_size: usize, stride: usize, pad: usize) -> usize {
    (input_size + pad * 2 - kernel_size) / stride + 1
}

// --- im2col / col2im ---

/// im2col: 4D 이미지 텐서를 2D 행렬로 변환 (순수 데이터 연산)
/// 입력 (N, C, H, W) → 출력 (N*OH*OW, C*KH*KW)
/// 각 행 = 커널 크기의 패치 하나를 펼친 것
fn im2col_data(
    x: &ArrayD<f64>, kh: usize, kw: usize,
    sh: usize, sw: usize, ph: usize, pw: usize,
) -> ArrayD<f64> {
    let shape = x.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let oh = get_conv_outsize(h, kh, sh, ph);
    let ow = get_conv_outsize(w, kw, sw, pw);

    // 패딩 적용
    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;
    let mut x_pad = ArrayD::zeros(ndarray::IxDyn(&[n, c, h_pad, w_pad]));
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    x_pad[[ni, ci, hi + ph, wi + pw]] = x[[ni, ci, hi, wi]];
                }
            }
        }
    }

    // im2col: 각 패치를 행으로 펼침
    let rows = n * oh * ow;
    let cols = c * kh * kw;
    let mut col = vec![0.0; rows * cols];

    for ni in 0..n {
        for i in 0..oh {
            for j in 0..ow {
                let row = ni * oh * ow + i * ow + j;
                let h_start = i * sh;
                let w_start = j * sw;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let col_idx = ci * kh * kw + ki * kw + kj;
                            col[row * cols + col_idx] =
                                x_pad[[ni, ci, h_start + ki, w_start + kj]];
                        }
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(ndarray::IxDyn(&[rows, cols]), col).unwrap()
}

/// col2im: im2col의 역연산 (2D 행렬 → 4D 텐서)
/// 입력 (N*OH*OW, C*KH*KW) → 출력 (N, C, H, W)
/// 겹치는 위치는 합산 (scatter-add)
fn col2im_data(
    col: &ArrayD<f64>, x_shape: &[usize],
    kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize,
) -> ArrayD<f64> {
    let (n, c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let oh = get_conv_outsize(h, kh, sh, ph);
    let ow = get_conv_outsize(w, kw, sw, pw);
    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;

    let mut x_pad = ArrayD::zeros(ndarray::IxDyn(&[n, c, h_pad, w_pad]));

    for ni in 0..n {
        for i in 0..oh {
            for j in 0..ow {
                let row = ni * oh * ow + i * ow + j;
                let h_start = i * sh;
                let w_start = j * sw;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let col_idx = ci * kh * kw + ki * kw + kj;
                            x_pad[[ni, ci, h_start + ki, w_start + kj]] +=
                                col[[row, col_idx]];
                        }
                    }
                }
            }
        }
    }

    // 패딩 제거
    if ph == 0 && pw == 0 {
        x_pad
    } else {
        let mut result = ArrayD::zeros(ndarray::IxDyn(&[n, c, h, w]));
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        result[[ni, ci, hi, wi]] = x_pad[[ni, ci, hi + ph, wi + pw]];
                    }
                }
            }
        }
        result
    }
}

/// Im2col Function: 역전파를 지원하는 im2col
struct Im2colFn {
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    ph: usize, pw: usize,
    x_shape: Vec<usize>,
}

impl Function for Im2colFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![im2col_data(&xs[0], self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy_data = gys[0].data();
        let gx = col2im_data(&gy_data, &self.x_shape, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw);
        vec![Variable::new(gx)]
    }
    fn name(&self) -> &str { "Im2col" }
}

/// Conv2d (simple): im2col + 행렬곱으로 합성곱 수행
struct Conv2dSimpleFn {
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    ph: usize, pw: usize,
    x_shape: Vec<usize>,
    w_shape: Vec<usize>,
    col: RefCell<ArrayD<f64>>, // forward에서 저장, backward에서 dw 계산에 사용
}

impl Function for Conv2dSimpleFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0]; // (N, C, H, W)
        let w = &xs[1]; // (OC, C, KH, KW)

        let n = self.x_shape[0];
        let oc = self.w_shape[0];
        let oh = get_conv_outsize(self.x_shape[2], self.kh, self.sh, self.ph);
        let ow = get_conv_outsize(self.x_shape[3], self.kw, self.sw, self.pw);

        // im2col: (N*OH*OW, C*KH*KW)
        let col = im2col_data(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw);

        // W → (OC, C*KH*KW)
        let ckk = self.w_shape[1] * self.kh * self.kw;
        let w_2d = w.to_shape((oc, ckk)).unwrap();

        // col @ W^T → (N*OH*OW, OC)
        let col_2d = col.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let y_2d = col_2d.dot(&w_2d.t()); // (N*OH*OW, OC)

        // (N*OH*OW, OC) → (N, OH, OW, OC) → (N, OC, OH, OW)
        let mut y = ArrayD::zeros(ndarray::IxDyn(&[n, oc, oh, ow]));
        for ni in 0..n {
            for oci in 0..oc {
                for hi in 0..oh {
                    for wi in 0..ow {
                        y[[ni, oci, hi, wi]] = y_2d[[ni * oh * ow + hi * ow + wi, oci]];
                    }
                }
            }
        }

        *self.col.borrow_mut() = col;
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = gys[0].data(); // (N, OC, OH, OW)
        let n = self.x_shape[0];
        let oc = self.w_shape[0];
        let ckk = self.w_shape[1] * self.kh * self.kw;
        let oh = get_conv_outsize(self.x_shape[2], self.kh, self.sh, self.ph);
        let ow = get_conv_outsize(self.x_shape[3], self.kw, self.sw, self.pw);

        // gy (N, OC, OH, OW) → gy_2d (N*OH*OW, OC)
        let mut gy_2d = ndarray::Array2::zeros((n * oh * ow, oc));
        for ni in 0..n {
            for oci in 0..oc {
                for hi in 0..oh {
                    for wi in 0..ow {
                        gy_2d[[ni * oh * ow + hi * ow + wi, oci]] = gy[[ni, oci, hi, wi]];
                    }
                }
            }
        }

        let col = self.col.borrow();
        let col_2d = col.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // W: (OC, C*KH*KW)
        let w_data = _xs[1].data();
        let w_2d = w_data.to_shape((oc, ckk)).unwrap();

        // dx: dcol = gy_2d @ W → (N*OH*OW, C*KH*KW) → col2im → (N,C,H,W)
        let dcol = gy_2d.dot(&w_2d);
        let dcol_dyn = dcol.into_dyn();
        let gx = col2im_data(&dcol_dyn, &self.x_shape, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw);

        // dw: col^T @ gy_2d → (C*KH*KW, OC) → transpose → (OC, C*KH*KW) → reshape (OC,C,KH,KW)
        let dw_2d = col_2d.t().dot(&gy_2d); // (C*KH*KW, OC)
        let dw_t = dw_2d.t().to_owned(); // (OC, C*KH*KW)
        let dw = ArrayD::from_shape_vec(ndarray::IxDyn(&self.w_shape), dw_t.into_raw_vec_and_offset().0).unwrap();

        vec![Variable::new(gx), Variable::new(dw)]
    }

    fn name(&self) -> &str { "Conv2d" }
}

/// im2col: 4D 텐서를 2D 행렬로 변환 (역전파 지원)
pub fn im2col(x: &Variable, kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> Variable {
    let x_shape = x.shape();
    Func::new(Im2colFn { kh, kw, sh, sw, ph, pw, x_shape }).call(&[x])
}

/// conv2d_simple: im2col + matmul 기반 2D 합성곱
pub fn conv2d_simple(x: &Variable, w: &Variable, b: Option<&Variable>, stride: usize, pad: usize) -> Variable {
    let x_shape = x.shape();
    let w_shape = w.shape();
    let kh = w_shape[2];
    let kw = w_shape[3];

    let y = Func::new(Conv2dSimpleFn {
        kh, kw, sh: stride, sw: stride, ph: pad, pw: pad,
        x_shape, w_shape,
        col: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
    }).call(&[x, w]);

    match b {
        Some(b) => &y + b,
        None => y,
    }
}

/// Embedding 순전파/역전파 함수
///
/// 순전파: 정수 인덱스 → 임베딩 벡터 룩업
///   W: (vocab_size, embed_dim) 가중치 테이블
///   idx: (N,) 또는 (N, T) 정수 인덱스 (f64로 전달, 내부에서 usize 변환)
///   출력: (..., embed_dim) — 인덱스 위치의 행 벡터를 가져옴
///
/// 역전파: scatter-add
///   gy: (..., embed_dim) 상위 기울기
///   gW: (vocab_size, embed_dim) — 각 인덱스 위치에 gy를 누적
///   gW[idx[i]] += gy[i]  (동일 인덱스면 합산)
struct EmbeddingFn {
    vocab_size: usize,
    idx_data: Vec<usize>,
    input_shape: Vec<usize>,
}

impl Function for EmbeddingFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let w = &xs[0]; // (vocab_size, embed_dim)
        let idx = &xs[1]; // 정수 인덱스 (f64로 인코딩)
        let embed_dim = w.shape()[1];

        // 인덱스에서 벡터를 룩업
        let indices: Vec<usize> = idx.iter().map(|&v| v as usize).collect();
        let n = indices.len();
        let mut out_data = Vec::with_capacity(n * embed_dim);
        for &i in &indices {
            let row = w.slice(ndarray::s![i, ..]);
            out_data.extend(row.iter());
        }

        // 출력 shape: input_shape + [embed_dim]
        let mut out_shape = idx.shape().to_vec();
        out_shape.push(embed_dim);
        let out = ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out_data).unwrap();
        vec![out]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = &gys[0]; // (..., embed_dim)
        let gy_data = gy.data();
        let embed_dim = gy_data.shape()[gy_data.ndim() - 1];

        // scatter-add: gW[idx[i]] += gy[i]
        let mut gw_data = ArrayD::zeros(ndarray::IxDyn(&[self.vocab_size, embed_dim]));
        let gy_2d = gy_data.view().into_shape_with_order(ndarray::IxDyn(&[self.idx_data.len(), embed_dim])).unwrap();
        for (i, &idx) in self.idx_data.iter().enumerate() {
            let row = gy_2d.slice(ndarray::s![i, ..]);
            let mut target = gw_data.slice_mut(ndarray::s![idx, ..]);
            target += &row;
        }

        // gW (W에 대한 기울기), gidx는 None (정수 인덱스는 미분 불가)
        let gw = Variable::new(gw_data);
        let gidx = Variable::new(ArrayD::zeros(ndarray::IxDyn(&self.input_shape)));
        vec![gw, gidx]
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

/// 임베딩 룩업: 정수 인덱스 → 벡터
/// W: (vocab_size, embed_dim), idx: 정수 인덱스 Variable
pub fn embedding(w: &Variable, idx: &Variable) -> Variable {
    let idx_data: Vec<usize> = idx.data().iter().map(|&v| v as usize).collect();
    let vocab_size = w.shape()[0];
    Func::new(EmbeddingFn {
        vocab_size,
        idx_data,
        input_shape: idx.shape(),
    }).call(&[w, idx])
}

/// 선형 변환: y = x @ W + b
/// matmul과 add의 조합이므로 역전파는 자동으로 처리됨
pub fn linear(x: &Variable, w: &Variable, b: Option<&Variable>) -> Variable {
    let t = matmul(x, w);
    match b {
        Some(b) => &t + b,
        None => t,
    }
}

/// 평균제곱오차: sum((x0 - x1)²) / n
pub fn mean_squared_error(x0: &Variable, x1: &Variable) -> Variable {
    let diff = x0 - x1;
    let n = diff.len();
    &sum(&diff.pow(2.0)) / (n as f64)
}

// --- 레이어 ---

/// Linear 레이어: y = x @ W + b
/// step43까지는 W1, b1, W2, b2를 개별 Variable로 관리했지만,
/// Linear 레이어로 묶으면:
///   - cleargrads() 한 번으로 W, b 기울기 모두 초기화
///   - params()로 모든 파라미터를 순회하며 업데이트
///   - W의 초기화를 레이어가 자동으로 처리 (lazy initialization)
pub struct Linear {
    out_size: usize,
    // 첫 forward 호출 전까지 None (lazy initialization)
    // 입력 데이터가 와야 in_size를 알 수 있기 때문
    w: RefCell<Option<Variable>>,
    b: Variable,
    // W 초기화용 난수 생성기 상태
    rng_state: Cell<u64>,
}

impl Linear {
    /// out_size만 지정하면 되고, in_size는 첫 forward 호출 시 자동 결정
    /// seed: W 초기화에 사용할 난수 시드 (재현 가능한 실험을 위해)
    pub fn new(out_size: usize, seed: u64) -> Self {
        Linear {
            out_size,
            w: RefCell::new(None),
            b: Variable::new(ArrayD::zeros(ndarray::IxDyn(&[out_size]))),
            rng_state: Cell::new(seed),
        }
    }

    // LCG 기반 균등분포 [0, 1) 난수 생성
    fn next_f64(&self) -> f64 {
        let state = self.rng_state.get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state.set(state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }

    // Box-Muller 변환으로 표준정규분포 난수 생성
    fn next_normal(&self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// 순전파: y = x @ W + b
    /// 첫 호출 시 x의 shape에서 in_size를 결정하고 W를 Xavier 초기화
    /// Xavier 초기화: randn(in, out) * sqrt(1/in)
    ///   입력 뉴런 수에 맞춰 분산을 조절하여 신호가 너무 커지거나 사라지는 것을 방지
    pub fn forward(&self, x: &Variable) -> Variable {
        if self.w.borrow().is_none() {
            let in_size = x.shape()[1];
            let scale = (1.0 / in_size as f64).sqrt();
            let w_data: Vec<f64> = (0..in_size * self.out_size)
                .map(|_| self.next_normal() * scale)
                .collect();
            *self.w.borrow_mut() = Some(Variable::new(
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[in_size, self.out_size]),
                    w_data,
                )
                .unwrap(),
            ));
        }
        linear(x, self.w.borrow().as_ref().unwrap(), Some(&self.b))
    }

    /// 모든 파라미터(W, b)의 기울기를 초기화
    /// step43: w1.cleargrad(); b1.cleargrad(); w2.cleargrad(); b2.cleargrad();
    /// step44: l1.cleargrads(); l2.cleargrads();  ← 레이어 단위로 간결해짐
    pub fn cleargrads(&self) {
        if let Some(w) = self.w.borrow().as_ref() {
            w.cleargrad();
        }
        self.b.cleargrad();
    }

    /// 모든 파라미터를 반환 (경사하강법에서 순회하며 업데이트할 때 사용)
    /// Variable의 Rc를 공유하므로 반환된 Variable을 수정하면 레이어 내부도 반영됨
    pub fn params(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        if let Some(w) = self.w.borrow().as_ref() {
            params.push(w.clone());
        }
        params.push(self.b.clone());
        params
    }
}

/// RNN (Recurrent Neural Network) 레이어
///
/// 순환 신경망의 핵심: 은닉 상태(hidden state)가 시간 스텝 간에 전달됨
///   h_new = tanh(x @ W_x + b + h @ W_h)
///
/// 시간 전개:
///   t=0: h₁ = tanh(x₀ @ W_x + b)           ← 초기 은닉 상태 없음
///   t=1: h₂ = tanh(x₁ @ W_x + b + h₁ @ W_h) ← h₁이 전달됨
///   t=2: h₃ = tanh(x₂ @ W_x + b + h₂ @ W_h) ← h₂가 전달됨
///
/// h는 Variable이므로 계산 그래프가 시간 스텝을 넘어 연결됨
/// → backward하면 시간을 거슬러 기울기가 전파됨 (BPTT)
/// → unchain_backward()로 절단하면 Truncated BPTT
pub struct RNN {
    x2h: Linear,
    w_h: RefCell<Option<Variable>>,
    h: RefCell<Option<Variable>>,
    hidden_size: usize,
    rng_state: Cell<u64>,
}

impl RNN {
    pub fn new(hidden_size: usize) -> Self {
        RNN {
            x2h: Linear::new(hidden_size, 99),
            w_h: RefCell::new(None),
            h: RefCell::new(None),
            hidden_size,
            rng_state: Cell::new(77),
        }
    }

    fn next_f64(&self) -> f64 {
        let state = self.rng_state.get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state.set(state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // w_h lazy init: 첫 호출 시 Xavier 초기화
        if self.w_h.borrow().is_none() {
            let scale = (1.0 / self.hidden_size as f64).sqrt();
            let w_data: Vec<f64> = (0..self.hidden_size * self.hidden_size)
                .map(|_| self.next_normal() * scale)
                .collect();
            *self.w_h.borrow_mut() = Some(Variable::new(
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[self.hidden_size, self.hidden_size]),
                    w_data,
                ).unwrap(),
            ));
        }

        let h_new = if self.h.borrow().is_none() {
            // 첫 스텝: h가 없으므로 x2h만 사용
            tanh(&self.x2h.forward(x))
        } else {
            // 이후 스텝: x2h(x) + h @ W_h
            let h = self.h.borrow().clone().unwrap();
            let w_h = self.w_h.borrow().as_ref().unwrap().clone();
            tanh(&(&self.x2h.forward(x) + &matmul(&h, &w_h)))
        };

        *self.h.borrow_mut() = Some(h_new.clone());
        h_new
    }

    pub fn cleargrads(&self) {
        self.x2h.cleargrads();
        if let Some(w) = self.w_h.borrow().as_ref() {
            w.cleargrad();
        }
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut params = self.x2h.params();
        if let Some(w) = self.w_h.borrow().as_ref() {
            params.push(w.clone());
        }
        params
    }
}

/// LSTM (Long Short-Term Memory) 레이어
///
/// RNN의 한계: 시퀀스가 길면 기울기 소실/폭발 (vanishing/exploding gradient)
/// LSTM은 셀 상태(cell state)와 게이트 메커니즘으로 이 문제를 해결
///
/// 4개의 게이트:
///   f = σ(x @ W_xf + h @ W_hf + b_f)   — forget: 이전 기억을 얼마나 잊을지
///   i = σ(x @ W_xi + h @ W_hi + b_i)   — input: 새 정보를 얼마나 기억할지
///   g = tanh(x @ W_xg + h @ W_hg + b_g) — candidate: 새로운 후보 기억
///   o = σ(x @ W_xo + h @ W_ho + b_o)   — output: 무엇을 출력할지
///
/// 상태 업데이트:
///   c_new = f ⊙ c + i ⊙ g              — 셀 상태: 선택적 기억/망각
///   h_new = o ⊙ tanh(c_new)             — 은닉 상태: 셀을 필터링한 출력
///
/// c는 "장기 기억", h는 "단기 기억(출력)" 역할
/// ⊙는 원소별 곱 (element-wise multiplication)
pub struct LSTM {
    // 입력 → 게이트 (bias 포함)
    x2f: Linear,
    x2i: Linear,
    x2o: Linear,
    x2g: Linear,
    // 은닉 → 게이트 (bias 없음, lazy init)
    w_hf: RefCell<Option<Variable>>,
    w_hi: RefCell<Option<Variable>>,
    w_ho: RefCell<Option<Variable>>,
    w_hg: RefCell<Option<Variable>>,
    // 상태
    h: RefCell<Option<Variable>>,
    c: RefCell<Option<Variable>>,
    hidden_size: usize,
    rng_state: Cell<u64>,
}

impl LSTM {
    pub fn new(hidden_size: usize) -> Self {
        LSTM {
            x2f: Linear::new(hidden_size, 200),
            x2i: Linear::new(hidden_size, 201),
            x2o: Linear::new(hidden_size, 202),
            x2g: Linear::new(hidden_size, 203),
            w_hf: RefCell::new(None),
            w_hi: RefCell::new(None),
            w_ho: RefCell::new(None),
            w_hg: RefCell::new(None),
            h: RefCell::new(None),
            c: RefCell::new(None),
            hidden_size,
            rng_state: Cell::new(300),
        }
    }

    fn next_f64(&self) -> f64 {
        let state = self.rng_state.get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state.set(state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Xavier 초기화된 (hidden_size, hidden_size) 가중치 생성
    fn init_w_h(&self) -> Variable {
        let scale = (1.0 / self.hidden_size as f64).sqrt();
        let w_data: Vec<f64> = (0..self.hidden_size * self.hidden_size)
            .map(|_| self.next_normal() * scale)
            .collect();
        Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[self.hidden_size, self.hidden_size]),
                w_data,
            ).unwrap(),
        )
    }

    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
        *self.c.borrow_mut() = None;
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        // h→gate 가중치 lazy init
        if self.w_hf.borrow().is_none() {
            *self.w_hf.borrow_mut() = Some(self.init_w_h());
            *self.w_hi.borrow_mut() = Some(self.init_w_h());
            *self.w_ho.borrow_mut() = Some(self.init_w_h());
            *self.w_hg.borrow_mut() = Some(self.init_w_h());
        }

        // 4개 게이트 계산
        let (f, i, o, g) = if self.h.borrow().is_none() {
            // 첫 스텝: h가 없으므로 x→gate만 사용
            (
                sigmoid(&self.x2f.forward(x)),
                sigmoid(&self.x2i.forward(x)),
                sigmoid(&self.x2o.forward(x)),
                tanh(&self.x2g.forward(x)),
            )
        } else {
            let h = self.h.borrow().clone().unwrap();
            let w_hf = self.w_hf.borrow().as_ref().unwrap().clone();
            let w_hi = self.w_hi.borrow().as_ref().unwrap().clone();
            let w_ho = self.w_ho.borrow().as_ref().unwrap().clone();
            let w_hg = self.w_hg.borrow().as_ref().unwrap().clone();
            (
                sigmoid(&(&self.x2f.forward(x) + &matmul(&h, &w_hf))),
                sigmoid(&(&self.x2i.forward(x) + &matmul(&h, &w_hi))),
                sigmoid(&(&self.x2o.forward(x) + &matmul(&h, &w_ho))),
                tanh(&(&self.x2g.forward(x) + &matmul(&h, &w_hg))),
            )
        };

        // 셀 상태 업데이트: c_new = f ⊙ c + i ⊙ g
        let c_new = if self.c.borrow().is_none() {
            &i * &g // 첫 스텝: forget할 것이 없음
        } else {
            let c = self.c.borrow().clone().unwrap();
            &(&f * &c) + &(&i * &g)
        };

        // 은닉 상태 업데이트: h_new = o ⊙ tanh(c_new)
        let h_new = &o * &tanh(&c_new);

        *self.h.borrow_mut() = Some(h_new.clone());
        *self.c.borrow_mut() = Some(c_new);
        h_new
    }

    pub fn cleargrads(&self) {
        self.x2f.cleargrads();
        self.x2i.cleargrads();
        self.x2o.cleargrads();
        self.x2g.cleargrads();
        for w in [&self.w_hf, &self.w_hi, &self.w_ho, &self.w_hg] {
            if let Some(w) = w.borrow().as_ref() {
                w.cleargrad();
            }
        }
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.x2f.params());
        params.extend(self.x2i.params());
        params.extend(self.x2o.params());
        params.extend(self.x2g.params());
        for w in [&self.w_hf, &self.w_hi, &self.w_ho, &self.w_hg] {
            if let Some(w) = w.borrow().as_ref() {
                params.push(w.clone());
            }
        }
        params
    }
}

/// Embedding 레이어: 정수 인덱스 → 밀집 벡터 변환
///
/// 자연어 처리의 기초: 단어(정수 ID)를 의미 있는 벡터 공간으로 매핑
///   "cat" → [0.2, -0.5, 0.8, ...]  (embed_dim 차원)
///   "dog" → [0.3, -0.4, 0.7, ...]  (의미적으로 가까운 벡터)
///
/// 내부적으로는 (vocab_size, embed_dim) 크기의 가중치 행렬 W에서
/// 입력 인덱스에 해당하는 행을 꺼내는 룩업 연산
/// W[3] = W의 3번째 행 → 인덱스 3에 대응하는 임베딩 벡터
///
/// one-hot @ W와 동일하지만, 실제로 one-hot 벡터를 만들지 않고 직접 인덱싱
/// → 메모리/연산 효율적
pub struct Embedding {
    w: Variable,
}

impl Embedding {
    /// vocab_size: 어휘 크기 (고유 토큰 수)
    /// embed_dim: 임베딩 벡터 차원
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        // Xavier 초기화
        let mut rng_state = seed;
        let scale = (1.0 / embed_dim as f64).sqrt();
        let w_data: Vec<f64> = (0..vocab_size * embed_dim)
            .map(|_| {
                // LCG
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u1 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                // Box-Muller
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                normal * scale
            })
            .collect();
        Embedding {
            w: Variable::new(
                ArrayD::from_shape_vec(ndarray::IxDyn(&[vocab_size, embed_dim]), w_data).unwrap(),
            ),
        }
    }

    /// idx: 정수 인덱스 Variable (값은 f64이지만 정수로 해석)
    pub fn forward(&self, idx: &Variable) -> Variable {
        embedding(&self.w, idx)
    }

    pub fn cleargrads(&self) {
        self.w.cleargrad();
    }

    pub fn params(&self) -> Vec<Variable> {
        vec![self.w.clone()]
    }
}

/// LayerNorm 레이어: 마지막 축을 따라 정규화
/// gamma (스케일)과 beta (시프트)를 학습 파라미터로 보유
/// gamma 초기값: 1 (정규화된 값을 그대로 유지)
/// beta 초기값: 0 (시프트 없음)
pub struct LayerNorm {
    gamma: Variable,
    beta: Variable,
    eps: f64,
}

impl LayerNorm {
    pub fn new(d: usize) -> Self {
        LayerNorm {
            gamma: Variable::new(ArrayD::ones(ndarray::IxDyn(&[d]))),
            beta: Variable::new(ArrayD::zeros(ndarray::IxDyn(&[d]))),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Variable) -> Variable {
        layer_norm(x, &self.gamma, &self.beta, self.eps)
    }

    pub fn cleargrads(&self) {
        self.gamma.cleargrad();
        self.beta.cleargrad();
    }

    pub fn params(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

/// SelfAttention: Multi-Head Self-Attention (causal / bidirectional 겸용)
///
/// 입력 (B, T, D)에서 Q, K, V를 생성하고, 멀티 헤드로 분할하여
/// scaled dot-product attention을 수행한 뒤 결합하여 (B, T, D)를 출력
///
/// use_causal_mask=true: GPT (미래 토큰 마스킹)
/// use_causal_mask=false: BERT (양방향 어텐션)
///
/// 데이터 흐름:
///   x (B,T,D) → Q,K,V 프로젝션 → (B,T,H,D_h) → transpose → (B,H,T,D_h)
///   → Q@K^T/√D_h → [causal mask] → softmax → dropout → @V
///   → transpose → (B,T,D) → 출력 프로젝션
pub struct SelfAttention {
    n_head: usize,
    n_embd: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    attn_dropout: f64,
    use_causal_mask: bool,
}

/// 하위 호환 alias: 기존 GPT 코드에서 CausalSelfAttention으로 사용
pub type CausalSelfAttention = SelfAttention;

impl SelfAttention {
    /// GPT용: causal mask 적용 (하위 호환)
    pub fn new(n_embd: usize, n_head: usize, dropout: f64, seed: u64) -> Self {
        Self::new_with_mask(n_embd, n_head, dropout, seed, true)
    }

    /// causal mask 여부를 직접 지정
    pub fn new_with_mask(n_embd: usize, n_head: usize, dropout: f64, seed: u64, use_causal_mask: bool) -> Self {
        assert!(n_embd % n_head == 0, "n_embd must be divisible by n_head");
        SelfAttention {
            n_head,
            n_embd,
            q_proj: Linear::new(n_embd, seed),
            k_proj: Linear::new(n_embd, seed.wrapping_add(1)),
            v_proj: Linear::new(n_embd, seed.wrapping_add(2)),
            out_proj: Linear::new(n_embd, seed.wrapping_add(3)),
            attn_dropout: dropout,
            use_causal_mask,
        }
    }

    /// x: (B, T, D) → (B, T, D)
    pub fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let d_head = d / self.n_head;

        // (B, T, D) → (B*T, D) — Linear은 2D 입력만 처리
        let x_2d = reshape(x, &[b * t, d]);

        // Q, K, V 프로젝션: (B*T, D) → (B*T, D)
        let q = self.q_proj.forward(&x_2d);
        let k = self.k_proj.forward(&x_2d);
        let v = self.v_proj.forward(&x_2d);

        // (B*T, D) → (B, T, H, D_head) → transpose → (B, H, T, D_head)
        let q = transpose_axes(&reshape(&q, &[b, t, self.n_head, d_head]), &[0, 2, 1, 3]);
        let k = transpose_axes(&reshape(&k, &[b, t, self.n_head, d_head]), &[0, 2, 1, 3]);
        let v = transpose_axes(&reshape(&v, &[b, t, self.n_head, d_head]), &[0, 2, 1, 3]);

        // Scaled dot-product attention
        // K^T: (B, H, D_head, T)
        let k_t = transpose_axes(&k, &[0, 1, 3, 2]);
        // Q @ K^T / √D_head: (B, H, T, T)
        let scores = &batched_matmul(&q, &k_t) / (d_head as f64).sqrt();
        // Causal mask (GPT) 또는 그대로 (BERT) → softmax → dropout
        let masked_scores = if self.use_causal_mask {
            causal_mask(&scores)
        } else {
            scores
        };
        let attn = dropout(&softmax(&masked_scores, -1), self.attn_dropout);
        // attn @ V: (B, H, T, T) @ (B, H, T, D_head) → (B, H, T, D_head)
        let out = batched_matmul(&attn, &v);

        // (B, H, T, D_head) → transpose → (B, T, H, D_head) → (B, T, D)
        let out = reshape(&transpose_axes(&out, &[0, 2, 1, 3]), &[b, t, d]);

        // 출력 프로젝션: (B*T, D) → (B*T, D) → (B, T, D)
        let out_2d = self.out_proj.forward(&reshape(&out, &[b * t, d]));
        reshape(&out_2d, &[b, t, d])
    }

    pub fn cleargrads(&self) {
        self.q_proj.cleargrads();
        self.k_proj.cleargrads();
        self.v_proj.cleargrads();
        self.out_proj.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.out_proj.params());
        p
    }
}

/// TransformerBlock: Pre-LN Transformer 블록 (GPT/BERT 겸용)
///
/// 구조:
///   x → LayerNorm → SelfAttention → Dropout → +x (residual)
///     → LayerNorm → FFN(Linear→GELU→Linear) → Dropout → +x (residual) → out
///
/// is_causal=true: GPT (미래 토큰 마스킹)
/// is_causal=false: BERT (양방향 어텐션)
///
/// Pre-LN: 정규화를 서브레이어 앞에 배치 (GPT-2/3 표준)
/// FFN: D → 4D → D로 확장 후 축소 (비선형 변환 용량 확보)
pub struct TransformerBlock {
    ln1: LayerNorm,
    attn: SelfAttention,
    ln2: LayerNorm,
    mlp_fc: Linear,      // D → 4D
    mlp_proj: Linear,    // 4D → D
    resid_dropout: f64,
}

impl TransformerBlock {
    /// GPT용: causal mask 적용 (하위 호환)
    pub fn new(n_embd: usize, n_head: usize, dropout: f64, seed: u64) -> Self {
        Self::new_with_causal(n_embd, n_head, dropout, seed, true)
    }

    /// is_causal: true=GPT(단방향), false=BERT(양방향)
    pub fn new_with_causal(n_embd: usize, n_head: usize, dropout: f64, seed: u64, is_causal: bool) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(n_embd),
            attn: SelfAttention::new_with_mask(n_embd, n_head, dropout, seed, is_causal),
            ln2: LayerNorm::new(n_embd),
            mlp_fc: Linear::new(4 * n_embd, seed.wrapping_add(100)),
            mlp_proj: Linear::new(n_embd, seed.wrapping_add(101)),
            resid_dropout: dropout,
        }
    }

    /// x: (B, T, D) → (B, T, D)
    pub fn forward(&self, x: &Variable) -> Variable {
        let shape = x.shape();
        let (b, t, d) = (shape[0], shape[1], shape[2]);

        // Sub-block 1: LayerNorm → Attention → Dropout → Residual
        let h = x + &dropout(&self.attn.forward(&self.ln1.forward(x)), self.resid_dropout);

        // Sub-block 2: LayerNorm → FFN → Dropout → Residual
        let normed = self.ln2.forward(&h);
        let normed_2d = reshape(&normed, &[b * t, d]);
        let ffn_out = self.mlp_proj.forward(&gelu(&self.mlp_fc.forward(&normed_2d)));
        let ffn_out = reshape(&ffn_out, &[b, t, d]);
        &h + &dropout(&ffn_out, self.resid_dropout)
    }

    pub fn cleargrads(&self) {
        self.ln1.cleargrads();
        self.attn.cleargrads();
        self.ln2.cleargrads();
        self.mlp_fc.cleargrads();
        self.mlp_proj.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.ln1.params());
        p.extend(self.attn.params());
        p.extend(self.ln2.params());
        p.extend(self.mlp_fc.params());
        p.extend(self.mlp_proj.params());
        p
    }
}

/// GPT: 완전한 Decoder-only Transformer 언어 모델
///
/// 구조:
///   token_ids (B,T) → Token Embedding + Position Embedding
///   → TransformerBlock × N → LayerNorm → Linear(vocab_size)
///   → logits (B, T, vocab_size)
///
/// 각 위치 t의 logit은 다음 토큰(t+1)의 확률 분포를 나타낸다.
/// 학습: softmax_cross_entropy로 다음 토큰 예측 loss 계산
/// 생성: 마지막 위치의 logit에서 argmax로 다음 토큰 선택, 자기회귀 반복
pub struct GPT {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    block_size: usize,
    n_embd: usize,
    vocab_size: usize,
}

impl GPT {
    /// block_size: 최대 시퀀스 길이
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        block_size: usize,
        dropout: f64,
        seed: u64,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            blocks.push(TransformerBlock::new(
                n_embd, n_head, dropout,
                seed.wrapping_add(1000 * (i as u64 + 1)),
            ));
        }
        GPT {
            token_emb: Embedding::new(vocab_size, n_embd, seed),
            pos_emb: Embedding::new(block_size, n_embd, seed.wrapping_add(1)),
            blocks,
            ln_f: LayerNorm::new(n_embd),
            lm_head: Linear::new(vocab_size, seed.wrapping_add(2)),
            block_size,
            n_embd,
            vocab_size,
        }
    }

    /// idx: (B, T) 정수 토큰 인덱스 → logits: (B, T, vocab_size)
    pub fn forward(&self, idx: &Variable) -> Variable {
        let shape = idx.shape();
        let (b, t) = (shape[0], shape[1]);

        // 토큰 임베딩: (B, T) → (B, T, D)
        let tok_emb = self.token_emb.forward(idx);

        // 위치 임베딩: [0, 1, ..., T-1] → (T, D) → broadcast to (B, T, D)
        let pos_idx = Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[t]),
                (0..t).map(|i| i as f64).collect(),
            ).unwrap()
        );
        let pos_emb = self.pos_emb.forward(&pos_idx); // (T, D)
        // (B, T, D) + (T, D) — ndarray가 자동 broadcast
        let mut x = &tok_emb + &pos_emb;

        // Transformer 블록 × N
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 최종 LayerNorm
        x = self.ln_f.forward(&x);

        // LM Head: (B, T, D) → (B*T, D) → (B*T, V) → (B, T, V)
        let x_2d = reshape(&x, &[b * t, self.n_embd]);
        let logits = self.lm_head.forward(&x_2d);
        reshape(&logits, &[b, t, self.vocab_size])
    }

    /// Greedy 자기회귀 생성
    /// start_tokens: 시작 토큰 시퀀스, max_new_tokens: 생성할 토큰 수
    pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
        let _guard = no_grad();
        let _test = test_mode();
        let mut tokens = start_tokens.to_vec();

        for _ in 0..max_new_tokens {
            // 최근 block_size 토큰만 사용 (위치 임베딩 범위 제한)
            let start = if tokens.len() > self.block_size {
                tokens.len() - self.block_size
            } else {
                0
            };
            let ctx = &tokens[start..];
            let t = ctx.len();

            let idx = Variable::new(
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[1, t]),
                    ctx.iter().map(|&v| v as f64).collect(),
                ).unwrap()
            );

            let logits = self.forward(&idx); // (1, T, V)
            let logits_data = logits.data();

            // 마지막 위치의 logit에서 argmax
            let last_t = t - 1;
            let mut best_tok = 0;
            let mut best_val = f64::NEG_INFINITY;
            for v in 0..self.vocab_size {
                let val = logits_data[[0, last_t, v]];
                if val > best_val {
                    best_val = val;
                    best_tok = v;
                }
            }
            tokens.push(best_tok);
        }

        tokens
    }

    pub fn cleargrads(&self) {
        self.token_emb.cleargrads();
        self.pos_emb.cleargrads();
        for block in &self.blocks {
            block.cleargrads();
        }
        self.ln_f.cleargrads();
        self.lm_head.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.token_emb.params());
        p.extend(self.pos_emb.params());
        for block in &self.blocks {
            p.extend(block.params());
        }
        p.extend(self.ln_f.params());
        p.extend(self.lm_head.params());
        p
    }
}

/// MaskedSoftmaxCrossEntropyFn: 마스크된 위치만 cross-entropy loss 계산
///
/// 전체 위치에 softmax를 계산하되, loss와 gradient는 mask=true인 위치만 반영.
/// 계산 그래프가 유지되어 backward가 정상 동작한다.
struct MaskedSoftmaxCrossEntropyFn {
    t: Vec<usize>,    // 정답 클래스 인덱스
    mask: Vec<bool>,  // true인 위치만 loss에 포함
}

impl Function for MaskedSoftmaxCrossEntropyFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0]; // (N, C)
        let n = x.shape()[0];
        let c = x.shape()[1];

        let masked_count = self.mask.iter().filter(|&&m| m).count();
        if masked_count == 0 {
            return vec![ndarray::arr0(0.0).into_dyn()];
        }

        // softmax (수치 안정 버전)
        let mut softmax = ArrayD::zeros(x.raw_dim());
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c { max_val = max_val.max(x[[i, j]]); }
            let mut sum_exp = 0.0;
            for j in 0..c {
                let e = (x[[i, j]] - max_val).exp();
                softmax[[i, j]] = e;
                sum_exp += e;
            }
            for j in 0..c { softmax[[i, j]] /= sum_exp; }
        }

        // cross-entropy: 마스크된 위치만
        let mut loss = 0.0;
        for i in 0..n {
            if self.mask[i] {
                let p = softmax[[i, self.t[i]]].max(1e-15);
                loss -= p.ln();
            }
        }
        loss /= masked_count as f64;

        vec![ndarray::arr0(loss).into_dyn()]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let x_data = xs[0].data();
        let n = x_data.shape()[0];
        let c = x_data.shape()[1];

        let masked_count = self.mask.iter().filter(|&&m| m).count();

        // softmax 재계산
        let mut softmax = ArrayD::zeros(x_data.raw_dim());
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c { max_val = max_val.max(x_data[[i, j]]); }
            let mut sum_exp = 0.0;
            for j in 0..c {
                let e = (x_data[[i, j]] - max_val).exp();
                softmax[[i, j]] = e;
                sum_exp += e;
            }
            for j in 0..c { softmax[[i, j]] /= sum_exp; }
        }

        // gradient: 마스크된 위치만 (softmax - one_hot) / M
        // 마스크 안 된 위치는 0
        let mut gx = ArrayD::zeros(x_data.raw_dim());
        for i in 0..n {
            if self.mask[i] {
                for j in 0..c {
                    gx[[i, j]] = softmax[[i, j]] / masked_count as f64;
                }
                gx[[i, self.t[i]]] -= 1.0 / masked_count as f64;
            }
        }

        let gy_val = gys[0].data().iter().next().copied().unwrap_or(1.0);
        let gx = gx.mapv(|v| v * gy_val);

        vec![Variable::new(gx)]
    }

    fn name(&self) -> &str { "MaskedSoftmaxCrossEntropy" }
}

/// masked_softmax_cross_entropy: 마스크된 위치만 loss 계산
///
/// logits: (N, C) 전체 위치의 logit
/// targets: 각 위치의 정답 클래스
/// mask: true인 위치만 loss에 포함 (MLM에서 [MASK] 위치)
///
/// loss = -mean_{i ∈ mask} log(softmax(logits_i)[targets_i])
pub fn masked_softmax_cross_entropy(logits: &Variable, targets: &[usize], mask: &[bool]) -> Variable {
    Func::new(MaskedSoftmaxCrossEntropyFn {
        t: targets.to_vec(),
        mask: mask.to_vec(),
    }).call(&[logits])
}

/// BERT: Bidirectional Encoder Transformer (MLM 언어 모델)
///
/// GPT와의 핵심 차이:
///   1. 양방향 어텐션 (causal mask 없음)
///   2. Segment embedding (문장 A=0 / B=1 구분)
///   3. MLM: [MASK] 위치의 원래 토큰을 예측
///
/// 구조:
///   token_ids (B,T) + segment_ids (B,T)
///   → Token Embedding + Position Embedding + Segment Embedding
///   → TransformerBlock(bidirectional) × N → LayerNorm → Linear(vocab_size)
///   → logits (B, T, vocab_size)
pub struct BERT {
    token_emb: Embedding,
    pos_emb: Embedding,
    segment_emb: Embedding,   // 2 → D (문장 A=0, B=1)
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    mlm_head: Linear,         // D → vocab_size
    max_seq_len: usize,
    n_embd: usize,
    vocab_size: usize,
}

impl BERT {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        max_seq_len: usize,
        dropout: f64,
        seed: u64,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            // is_causal=false → 양방향 어텐션
            blocks.push(TransformerBlock::new_with_causal(
                n_embd, n_head, dropout,
                seed.wrapping_add(1000 * (i as u64 + 1)),
                false,
            ));
        }
        BERT {
            token_emb: Embedding::new(vocab_size, n_embd, seed),
            pos_emb: Embedding::new(max_seq_len, n_embd, seed.wrapping_add(1)),
            segment_emb: Embedding::new(2, n_embd, seed.wrapping_add(2)),
            blocks,
            ln_f: LayerNorm::new(n_embd),
            mlm_head: Linear::new(vocab_size, seed.wrapping_add(3)),
            max_seq_len,
            n_embd,
            vocab_size,
        }
    }

    /// token_ids: (B, T), segment_ids: (B, T) → hidden: (B, T, D)
    /// MLM head 이전의 hidden states를 반환 (문장 임베딩 등에서 사용)
    pub fn forward_hidden(&self, token_ids: &Variable, segment_ids: &Variable) -> Variable {
        let shape = token_ids.shape();
        let t = shape[shape.len() - 1];

        // 토큰 임베딩: (B, T) → (B, T, D)
        let tok_emb = self.token_emb.forward(token_ids);

        // 위치 임베딩: [0, 1, ..., T-1] → (T, D) → broadcast to (B, T, D)
        let pos_idx = Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[t]),
                (0..t).map(|i| i as f64).collect(),
            ).unwrap()
        );
        let pos_emb = self.pos_emb.forward(&pos_idx); // (T, D)

        // 세그먼트 임베딩: (B, T) → (B, T, D)
        let seg_emb = self.segment_emb.forward(segment_ids);

        // 3개 임베딩 합산
        let mut x = &(&tok_emb + &pos_emb) + &seg_emb;

        // Transformer 블록 × N (양방향 어텐션)
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 최종 LayerNorm → (B, T, D)
        self.ln_f.forward(&x)
    }

    /// token_ids: (B, T), segment_ids: (B, T) → logits: (B, T, vocab_size)
    pub fn forward(&self, token_ids: &Variable, segment_ids: &Variable) -> Variable {
        let hidden = self.forward_hidden(token_ids, segment_ids); // (B, T, D)
        let shape = hidden.shape();
        let (b, t) = (shape[0], shape[1]);

        // MLM Head: (B, T, D) → (B*T, D) → (B*T, V) → (B, T, V)
        let x_2d = reshape(&hidden, &[b * t, self.n_embd]);
        let logits = self.mlm_head.forward(&x_2d);
        reshape(&logits, &[b, t, self.vocab_size])
    }

    pub fn cleargrads(&self) {
        self.token_emb.cleargrads();
        self.pos_emb.cleargrads();
        self.segment_emb.cleargrads();
        for block in &self.blocks {
            block.cleargrads();
        }
        self.ln_f.cleargrads();
        self.mlm_head.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.token_emb.params());
        p.extend(self.pos_emb.params());
        p.extend(self.segment_emb.params());
        for block in &self.blocks {
            p.extend(block.params());
        }
        p.extend(self.ln_f.params());
        p.extend(self.mlm_head.params());
        p
    }
}

/// Negative Sampling Loss: Skip-gram의 효율적 손실 함수
///
/// 전체 어휘에 softmax 대신, 정답 1개 + 부정 K개만으로 이진 분류:
///   L = -log σ(u_pos · v) - Σ_k log σ(-u_neg_k · v)
///
/// 입력: v_center (N, D), u_all (N*(1+K), D)
/// labels: [+1, -1, -1, ..., -1] per sample (1+K 반복 N번)
///
/// Gradient: d/d(dot) = (σ(dot) - label_01) / N
///   정답(label_01=1): σ(dot) - 1 → 확률이 높을수록 gradient 작음
///   부정(label_01=0): σ(dot) → 확률이 낮을수록 gradient 작음
struct NegativeSamplingLossFn {
    labels: Vec<f64>,       // +1.0 (정답) 또는 -1.0 (부정)
    num_negative: usize,    // K
    batch_size: usize,      // N
    embed_dim: usize,       // D
}

impl Function for NegativeSamplingLossFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let v = &xs[0];  // (N, D) — 중심 단어 임베딩
        let u = &xs[1];  // (N*(1+K), D) — 정답+부정 단어 임베딩

        let n = self.batch_size;
        let k1 = 1 + self.num_negative;
        let d = self.embed_dim;

        let mut loss = 0.0;
        for i in 0..n {
            for j in 0..k1 {
                let idx = i * k1 + j;
                let mut dot = 0.0;
                for dd in 0..d {
                    dot += v[[i, dd]] * u[[idx, dd]];
                }
                // log_sigmoid(label * dot) — 수치 안정 버전
                // log σ(x) = min(0, x) - log(1 + exp(-|x|))
                let x = self.labels[idx] * dot;
                let log_sig = x.min(0.0) - (1.0 + (-x.abs()).exp()).ln();
                loss -= log_sig;
            }
        }
        loss /= n as f64;

        vec![ndarray::arr0(loss).into_dyn()]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let v_data = xs[0].data();  // (N, D)
        let u_data = xs[1].data();  // (N*(1+K), D)
        let gy_val = gys[0].data().iter().next().copied().unwrap_or(1.0);

        let n = self.batch_size;
        let k1 = 1 + self.num_negative;
        let d = self.embed_dim;

        let mut gv = ArrayD::zeros(v_data.raw_dim());
        let mut gu = ArrayD::zeros(u_data.raw_dim());

        for i in 0..n {
            for j in 0..k1 {
                let idx = i * k1 + j;
                let mut dot = 0.0;
                for dd in 0..d {
                    dot += v_data[[i, dd]] * u_data[[idx, dd]];
                }
                let sig = 1.0 / (1.0 + (-dot).exp());
                let label_01 = if self.labels[idx] > 0.0 { 1.0 } else { 0.0 };
                let grad_dot = (sig - label_01) / n as f64 * gy_val;

                for dd in 0..d {
                    gv[[i, dd]] += grad_dot * u_data[[idx, dd]];
                    gu[[idx, dd]] += grad_dot * v_data[[i, dd]];
                }
            }
        }

        vec![Variable::new(gv), Variable::new(gu)]
    }

    fn name(&self) -> &str { "NegativeSamplingLoss" }
}

/// Negative Sampling Loss 함수
///
/// v_center: (N, D) 중심 단어 임베딩
/// u_all: (N*(1+K), D) 정답+부정 단어 임베딩
/// labels: +1.0(정답) / -1.0(부정), 길이 N*(1+K)
/// num_negative: K
pub fn negative_sampling_loss(
    v_center: &Variable,
    u_all: &Variable,
    labels: &[f64],
    num_negative: usize,
) -> Variable {
    let batch_size = v_center.shape()[0];
    let embed_dim = v_center.shape()[1];
    Func::new(NegativeSamplingLossFn {
        labels: labels.to_vec(),
        num_negative,
        batch_size,
        embed_dim,
    }).call(&[v_center, u_all])
}

/// Word2Vec: Skip-gram + Negative Sampling 모델
///
/// 두 개의 Embedding 레이어:
///   W_in: 중심 단어 임베딩 (학습 후 단어 벡터로 사용)
///   W_out: 문맥 단어 임베딩 (학습 보조, 보통 버림)
///
/// 학습: center_word로 context_word를 예측하되,
/// K개의 부정 샘플과 구분하는 이진 분류로 효율화
pub struct Word2Vec {
    w_in: Embedding,
    w_out: Embedding,
    vocab_size: usize,
    embed_dim: usize,
}

impl Word2Vec {
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        Word2Vec {
            w_in: Embedding::new(vocab_size, embed_dim, seed),
            w_out: Embedding::new(vocab_size, embed_dim, seed.wrapping_add(1)),
            vocab_size,
            embed_dim,
        }
    }

    /// Skip-gram forward
    /// center_ids: (N,) 중심 단어 인덱스
    /// context_ids: (N,) 정답 문맥 단어 인덱스
    /// negative_ids: (N, K) 부정 샘플 인덱스
    /// 반환: 스칼라 loss
    pub fn forward(
        &self,
        center_ids: &Variable,
        context_ids: &Variable,
        negative_ids: &Variable,
    ) -> Variable {
        let n = center_ids.shape()[0];
        let k = if negative_ids.shape().len() == 2 {
            negative_ids.shape()[1]
        } else {
            negative_ids.shape()[0] / n
        };

        // 중심 단어 임베딩: (N,) → (N, D)
        let v_center = self.w_in.forward(center_ids);

        // 정답 + 부정 단어 인덱스를 하나로 합침: (N*(1+K),)
        let ctx_data = context_ids.data();
        let neg_data = negative_ids.data();
        let mut all_ids_data = Vec::with_capacity(n * (1 + k));
        for i in 0..n {
            all_ids_data.push(ctx_data[i]);
            for j in 0..k {
                all_ids_data.push(neg_data[[i, j]]);
            }
        }
        let all_ids = Variable::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[n * (1 + k)]), all_ids_data).unwrap()
        );

        // 모든 타겟 임베딩: (N*(1+K),) → (N*(1+K), D)
        let u_all = self.w_out.forward(&all_ids);

        // 라벨: [+1, -1, -1, ..., -1] per sample
        let labels: Vec<f64> = (0..n).flat_map(|_| {
            std::iter::once(1.0).chain(std::iter::repeat_n(-1.0, k))
        }).collect();

        negative_sampling_loss(&v_center, &u_all, &labels, k)
    }

    pub fn cleargrads(&self) {
        self.w_in.cleargrads();
        self.w_out.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = self.w_in.params();
        p.extend(self.w_out.params());
        p
    }

    /// 학습된 단어 벡터 반환 (W_in의 가중치)
    pub fn get_word_vectors(&self) -> ArrayD<f64> {
        self.w_in.params()[0].data()
    }
}

/// NT-Xent (Normalized Temperature-scaled Cross-Entropy) Loss
///
/// SimCLR/CLIP에서 사용하는 대칭 contrastive loss:
///   1. L2 정규화: ẑ = z / ‖z‖
///   2. 유사도 행렬: S_ij = ẑ_a_i · ẑ_b_j / τ
///   3. 대칭 CE: L = ½[CE(S, diag) + CE(S^T, diag)]
///
/// 양의 쌍(대각선)은 가깝게, 나머지(배치 내 부정 쌍)는 멀리 밀어냄
struct NTXentLossFn {
    temperature: f64,
    batch_size: usize,
    embed_dim: usize,
}

impl Function for NTXentLossFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let z_a = &xs[0]; // (N, D)
        let z_b = &xs[1]; // (N, D)
        let n = self.batch_size;
        let d = self.embed_dim;

        // 1. L2 정규화 (행별)
        let mut z_a_norm = ArrayD::zeros(z_a.raw_dim());
        let mut z_b_norm = ArrayD::zeros(z_b.raw_dim());
        for i in 0..n {
            let mut norm_a = 0.0f64;
            let mut norm_b = 0.0f64;
            for j in 0..d {
                norm_a += z_a[[i, j]] * z_a[[i, j]];
                norm_b += z_b[[i, j]] * z_b[[i, j]];
            }
            norm_a = norm_a.sqrt().max(1e-12);
            norm_b = norm_b.sqrt().max(1e-12);
            for j in 0..d {
                z_a_norm[[i, j]] = z_a[[i, j]] / norm_a;
                z_b_norm[[i, j]] = z_b[[i, j]] / norm_b;
            }
        }

        // 2. 유사도 행렬: S = ẑ_a @ ẑ_b^T / τ → (N, N)
        let mut sim = ArrayD::zeros(ndarray::IxDyn(&[n, n]));
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += z_a_norm[[i, k]] * z_b_norm[[j, k]];
                }
                sim[[i, j]] = dot / self.temperature;
            }
        }

        // 3. 대칭 Cross-Entropy: target = diagonal (i→i)
        //    L = ½[CE(S, diag) + CE(S^T, diag)]
        let mut loss = 0.0;

        // CE(S, diag): 각 행 i에서 target = i
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..n { max_val = max_val.max(sim[[i, j]]); }
            let mut sum_exp = 0.0;
            for j in 0..n { sum_exp += (sim[[i, j]] - max_val).exp(); }
            let log_softmax_ii = sim[[i, i]] - max_val - sum_exp.ln();
            loss -= log_softmax_ii;
        }

        // CE(S^T, diag): S 전치의 각 행 j에서 target = j
        for j in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..n { max_val = max_val.max(sim[[i, j]]); }
            let mut sum_exp = 0.0;
            for i in 0..n { sum_exp += (sim[[i, j]] - max_val).exp(); }
            let log_softmax_jj = sim[[j, j]] - max_val - sum_exp.ln();
            loss -= log_softmax_jj;
        }

        loss /= (2 * n) as f64;

        vec![ndarray::arr0(loss).into_dyn()]
    }

    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let z_a_data = xs[0].data(); // (N, D)
        let z_b_data = xs[1].data(); // (N, D)
        let gy_val = gys[0].data().iter().next().copied().unwrap_or(1.0);
        let n = self.batch_size;
        let d = self.embed_dim;

        // 1. L2 정규화
        let mut z_a_hat = ArrayD::zeros(z_a_data.raw_dim());
        let mut z_b_hat = ArrayD::zeros(z_b_data.raw_dim());
        let mut norms_a = vec![0.0f64; n];
        let mut norms_b = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..d {
                norms_a[i] += z_a_data[[i, j]] * z_a_data[[i, j]];
                norms_b[i] += z_b_data[[i, j]] * z_b_data[[i, j]];
            }
            norms_a[i] = norms_a[i].sqrt().max(1e-12);
            norms_b[i] = norms_b[i].sqrt().max(1e-12);
            for j in 0..d {
                z_a_hat[[i, j]] = z_a_data[[i, j]] / norms_a[i];
                z_b_hat[[i, j]] = z_b_data[[i, j]] / norms_b[i];
            }
        }

        // 2. 유사도 행렬 S = ẑ_a @ ẑ_b^T / τ
        let mut sim = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..d { dot += z_a_hat[[i, k]] * z_b_hat[[j, k]]; }
                sim[i][j] = dot / self.temperature;
            }
        }

        // 3. dL/dS 계산
        //    dL/dS = ½[(softmax_row(S) - I)/N + ((softmax_col(S) - I)/N)^T]
        let mut ds = vec![vec![0.0f64; n]; n];

        // softmax(S) 행별 — CE(S, diag) 기울기
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..n { max_val = max_val.max(sim[i][j]); }
            let mut sum_exp = 0.0;
            let mut exps = vec![0.0; n];
            for j in 0..n {
                exps[j] = (sim[i][j] - max_val).exp();
                sum_exp += exps[j];
            }
            for j in 0..n {
                let softmax_ij = exps[j] / sum_exp;
                let target = if i == j { 1.0 } else { 0.0 };
                ds[i][j] += 0.5 * (softmax_ij - target) / n as f64;
            }
        }

        // softmax(S^T) 열별 — CE(S^T, diag) 기울기
        for j in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..n { max_val = max_val.max(sim[i][j]); }
            let mut sum_exp = 0.0;
            let mut exps = vec![0.0; n];
            for i in 0..n {
                exps[i] = (sim[i][j] - max_val).exp();
                sum_exp += exps[i];
            }
            for i in 0..n {
                let softmax_ji = exps[i] / sum_exp;
                let target = if i == j { 1.0 } else { 0.0 };
                // S^T의 행 j, 열 i → S의 행 i, 열 j (전치)
                ds[i][j] += 0.5 * (softmax_ji - target) / n as f64;
            }
        }

        // 4. dL/dẑ_a = dS · ẑ_b / τ, dL/dẑ_b = dS^T · ẑ_a / τ
        let mut g_a_hat: ArrayD<f64> = ArrayD::zeros(z_a_data.raw_dim());
        let mut g_b_hat: ArrayD<f64> = ArrayD::zeros(z_b_data.raw_dim());
        for i in 0..n {
            for j in 0..n {
                let ds_ij = ds[i][j] / self.temperature;
                for k in 0..d {
                    g_a_hat[[i, k]] += ds_ij * z_b_hat[[j, k]];
                    g_b_hat[[j, k]] += ds_ij * z_a_hat[[i, k]];
                }
            }
        }

        // 5. L2 norm 역전파: dL/dz = (g - ẑ(g·ẑ)) / ‖z‖
        let mut g_a = ArrayD::zeros(z_a_data.raw_dim());
        let mut g_b = ArrayD::zeros(z_b_data.raw_dim());
        for i in 0..n {
            let mut dot_a = 0.0;
            let mut dot_b = 0.0;
            for k in 0..d {
                dot_a += g_a_hat[[i, k]] * z_a_hat[[i, k]];
                dot_b += g_b_hat[[i, k]] * z_b_hat[[i, k]];
            }
            for k in 0..d {
                g_a[[i, k]] = (g_a_hat[[i, k]] - z_a_hat[[i, k]] * dot_a) / norms_a[i] * gy_val;
                g_b[[i, k]] = (g_b_hat[[i, k]] - z_b_hat[[i, k]] * dot_b) / norms_b[i] * gy_val;
            }
        }

        vec![Variable::new(g_a), Variable::new(g_b)]
    }

    fn name(&self) -> &str { "NTXentLoss" }
}

/// NT-Xent (InfoNCE) Loss: 대칭 contrastive loss
///
/// z_a: (N, D) — 앵커 임베딩
/// z_b: (N, D) — 양의 쌍 임베딩
/// temperature: τ (보통 0.05~0.5, 작을수록 hard negative에 집중)
///
/// L = ½[CE(cos(z_a, z_b)/τ, diag) + CE(cos(z_a, z_b)^T/τ, diag)]
pub fn nt_xent_loss(z_a: &Variable, z_b: &Variable, temperature: f64) -> Variable {
    let batch_size = z_a.shape()[0];
    let embed_dim = z_a.shape()[1];
    Func::new(NTXentLossFn {
        temperature,
        batch_size,
        embed_dim,
    }).call(&[z_a, z_b])
}

/// Sentence Embedding: BERT 인코더 + Mean Pooling + Projection
///
/// SBERT (Reimers & Gurevych, 2019)에서 영감:
/// 1. BERT 인코더로 토큰별 hidden states 생성 (B, T, D)
/// 2. Mean Pooling으로 문장 벡터 추출 (B, D)
/// 3. Projection head로 contrastive 학습 공간에 매핑 (B, P)
///
/// 학습: NT-Xent loss로 유사 문장 쌍은 가깝게, 다른 문장은 멀리
/// 추론: projection 전 hidden (encode)을 문장 벡터로 사용
pub struct SentenceEmbedding {
    token_emb: Embedding,
    pos_emb: Embedding,
    segment_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    projection: Linear,
    n_embd: usize,
    proj_dim: usize,
}

impl SentenceEmbedding {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        max_seq_len: usize,
        proj_dim: usize,
        dropout: f64,
        seed: u64,
    ) -> Self {
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            blocks.push(TransformerBlock::new_with_causal(
                n_embd, n_head, dropout,
                seed.wrapping_add(1000 * (i as u64 + 1)),
                false, // 양방향 어텐션
            ));
        }
        SentenceEmbedding {
            token_emb: Embedding::new(vocab_size, n_embd, seed),
            pos_emb: Embedding::new(max_seq_len, n_embd, seed.wrapping_add(1)),
            segment_emb: Embedding::new(2, n_embd, seed.wrapping_add(2)),
            blocks,
            ln_f: LayerNorm::new(n_embd),
            projection: Linear::new(proj_dim, seed.wrapping_add(3)),
            n_embd,
            proj_dim,
        }
    }

    /// token_ids: (B, T), segment_ids: (B, T) → (B, proj_dim)
    /// 학습용: projection head 포함
    pub fn forward(&self, token_ids: &Variable, segment_ids: &Variable) -> Variable {
        let hidden = self.encode(token_ids, segment_ids); // (B, D)
        let b = hidden.shape()[0];
        let h2d = reshape(&hidden, &[b, self.n_embd]);
        self.projection.forward(&h2d) // (B, proj_dim)
    }

    /// token_ids: (B, T), segment_ids: (B, T) → (B, D)
    /// 추론용: mean pooling 후 projection 전의 hidden states
    pub fn encode(&self, token_ids: &Variable, segment_ids: &Variable) -> Variable {
        let shape = token_ids.shape();
        let t = shape[shape.len() - 1];

        // 3개 임베딩 합산
        let tok_emb = self.token_emb.forward(token_ids);
        let pos_idx = Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[t]),
                (0..t).map(|i| i as f64).collect(),
            ).unwrap()
        );
        let pos_emb = self.pos_emb.forward(&pos_idx);
        let seg_emb = self.segment_emb.forward(segment_ids);
        let mut x = &(&tok_emb + &pos_emb) + &seg_emb; // (B, T, D)

        // Transformer 블록
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // LayerNorm → (B, T, D)
        x = self.ln_f.forward(&x);

        // Mean Pooling: (B, T, D) → sum over T → (B, D) → / T
        let summed = sum_with(&x, Some(1), false); // (B, D)
        let t_inv = Variable::new(ndarray::arr0(1.0 / t as f64).into_dyn());
        &summed * &t_inv
    }

    pub fn cleargrads(&self) {
        self.token_emb.cleargrads();
        self.pos_emb.cleargrads();
        self.segment_emb.cleargrads();
        for block in &self.blocks {
            block.cleargrads();
        }
        self.ln_f.cleargrads();
        self.projection.cleargrads();
    }

    pub fn params(&self) -> Vec<Variable> {
        let mut p = Vec::new();
        p.extend(self.token_emb.params());
        p.extend(self.pos_emb.params());
        p.extend(self.segment_emb.params());
        for block in &self.blocks {
            p.extend(block.params());
        }
        p.extend(self.ln_f.params());
        p.extend(self.projection.params());
        p
    }
}

/// Model 트레잇: 여러 레이어를 하나의 모델로 묶어서 관리
/// step44에서는 l1.cleargrads(), l2.cleargrads()를 각각 호출했지만,
/// Model로 묶으면 model.cleargrads() 한 번으로 모든 레이어의 기울기를 초기화
///
/// 사용법: 사용자가 자신의 모델 구조체를 만들고 Model 트레잇을 구현
///   struct MyModel { l1: Linear, l2: Linear }
///   impl Model for MyModel {
///       fn forward(&self, x: &Variable) -> Variable { ... }
///       fn layers(&self) -> Vec<&Linear> { vec![&self.l1, &self.l2] }
///   }
pub trait Model {
    /// 순전파 (모델의 네트워크 구조를 정의)
    fn forward(&self, x: &Variable) -> Variable;

    /// 모델이 포함하는 모든 레이어를 반환
    /// Python에서는 __setattr__로 자동 등록하지만
    /// Rust에서는 사용자가 명시적으로 나열
    fn layers(&self) -> Vec<&Linear>;

    /// 모든 레이어의 모든 파라미터 기울기를 초기화
    /// step44: l1.cleargrads(); l2.cleargrads();
    /// step45: model.cleargrads();  ← 한 번으로 끝
    fn cleargrads(&self) {
        for l in self.layers() {
            l.cleargrads();
        }
    }

    /// 모든 레이어의 모든 파라미터를 반환
    /// step44: for l in [&l1, &l2] { for p in l.params() { ... } }
    /// step45: for p in model.params() { ... }  ← 모델 단위로 순회
    fn params(&self) -> Vec<Variable> {
        self.layers().iter().flat_map(|l| l.params()).collect()
    }

    /// 모든 파라미터를 바이너리 파일로 저장
    /// Python의 model.save_weights('file.npz')에 해당
    ///
    /// 바이너리 포맷 (little-endian):
    ///   파라미터 수 (u32)
    ///   각 파라미터: ndim (u32) + shape (u32 × ndim) + data (f64 × 원소 수)
    ///
    /// params()의 순서가 결정적이므로 load_weights와 1:1 대응:
    ///   layers[0].W, layers[0].b, layers[1].W, layers[1].b, ...
    fn save_weights(&self, path: &str) {
        use std::io::Write;
        let params = self.params();
        let mut file = std::fs::File::create(path).expect("failed to create weight file");

        // 파라미터 개수
        file.write_all(&(params.len() as u32).to_le_bytes()).unwrap();

        for p in &params {
            let data = p.data();
            let shape = data.shape();

            // ndim + shape
            file.write_all(&(shape.len() as u32).to_le_bytes()).unwrap();
            for &dim in shape {
                file.write_all(&(dim as u32).to_le_bytes()).unwrap();
            }

            // f64 데이터 (raw bytes)
            for &val in data.iter() {
                file.write_all(&val.to_le_bytes()).unwrap();
            }
        }
    }

    /// 바이너리 파일에서 파라미터를 로드하여 모델에 복원
    /// Python의 model.load_weights('file.npz')에 해당
    ///
    /// Rc<RefCell<>>로 공유되므로 set_data()가 레이어 원본도 갱신:
    ///   params()[i].set_data(loaded_data)
    ///   → layers[k].W 또는 layers[k].b의 내부 데이터가 교체됨
    ///
    /// 주의: lazy init 전(forward 미호출)에는 W가 None이라 params()에 포함 안 됨.
    ///       저장 시점과 로드 시점의 파라미터 수가 일치해야 함.
    fn load_weights(&self, path: &str) {
        let bytes = std::fs::read(path).expect("failed to read weight file");
        let mut offset = 0;

        // 파라미터 개수 읽기
        let num_params = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let params = self.params();
        assert_eq!(
            params.len(),
            num_params,
            "parameter count mismatch: model has {}, file has {}",
            params.len(),
            num_params
        );

        for p in &params {
            // ndim
            let ndim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            // shape
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                let dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                offset += 4;
                shape.push(dim);
            }

            // f64 데이터
            let num_elements: usize = shape.iter().product();
            let mut data = Vec::with_capacity(num_elements);
            for _ in 0..num_elements {
                let val = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                offset += 8;
                data.push(val);
            }

            let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data).unwrap();
            p.set_data(arr);
        }
    }
}

/// MLP (Multi-Layer Perceptron): 범용 다층 신경망
/// step45의 TwoLayerNet은 2층 고정이었지만,
/// MLP는 임의의 층 수를 지원: MLP::new(&[10, 5, 1]) → 3층
/// 마지막 층을 제외한 모든 층에 활성화 함수(기본: sigmoid) 적용
///
/// 예시) MLP::new(&[10, 1]):
///   x → Linear(10) → sigmoid → Linear(1) → 출력
///
/// 예시) MLP::new(&[20, 10, 1]):
///   x → Linear(20) → sigmoid → Linear(10) → sigmoid → Linear(1) → 출력
pub struct MLP {
    layers: Vec<Linear>,
    activation: fn(&Variable) -> Variable,
}

impl MLP {
    /// sizes: 각 층의 출력 크기
    /// 예) &[10, 1] → Linear(?→10) → sigmoid → Linear(10→1)
    pub fn new(sizes: &[usize]) -> Self {
        let layers = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| Linear::new(size, 42 + i as u64))
            .collect();
        MLP {
            layers,
            activation: sigmoid,
        }
    }
}

impl Model for MLP {
    fn forward(&self, x: &Variable) -> Variable {
        let last = self.layers.len() - 1;
        let mut h = x.clone(); // Rc clone (데이터 복사 아님)
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h);
            // 마지막 층에는 활성화 함수를 적용하지 않음
            // 회귀: 출력 범위 제한 없이 실수값 출력
            // 분류: 별도의 softmax 등을 적용하기 위해
            if i < last {
                h = (self.activation)(&h);
            }
        }
        h
    }

    fn layers(&self) -> Vec<&Linear> {
        self.layers.iter().collect()
    }
}

// --- 옵티마이저 ---

/// SGD (Stochastic Gradient Descent) 옵티마이저
/// step45까지는 파라미터 업데이트를 직접 작성:
///   for p in model.params() {
///       p.set_data(&p.data() - &grad.mapv(|v| v * lr));
///   }
/// SGD 옵티마이저가 이 로직을 캡슐화:
///   optimizer.update();  ← 한 줄로 끝
///
/// 업데이트 규칙: p ← p - lr × ∂L/∂p
pub struct SGD<'a> {
    lr: f64,
    target: Option<&'a dyn Model>,
}

impl<'a> SGD<'a> {
    pub fn new(lr: f64) -> Self {
        SGD { lr, target: None }
    }

    /// 모델과 연결. Python의 optimizer.setup(model)에 해당.
    /// 체이닝 가능: SGD::new(lr).setup(&model)
    pub fn setup(mut self, model: &'a dyn Model) -> Self {
        self.target = Some(model);
        self
    }

    /// 모든 파라미터를 SGD 규칙으로 업데이트
    pub fn update(&self) {
        let model = self.target.expect("call setup() before update()");
        for p in model.params() {
            let grad = p.grad().unwrap();
            p.set_data(&p.data() - &grad.mapv(|v| v * self.lr));
        }
    }
}

/// Adam 옵티마이저 (Kingma & Ba, 2015)
///
/// SGD는 모든 파라미터에 동일한 학습률을 적용하지만,
/// Adam은 각 파라미터별로 적응적 학습률을 사용한다.
///
/// 업데이트 규칙:
///   m ← β₁·m + (1-β₁)·grad        (1차 모멘트: 기울기의 지수 이동평균)
///   v ← β₂·v + (1-β₂)·grad²       (2차 모멘트: 기울기 제곱의 이동평균)
///   lr_t = lr × √(1-β₂ᵗ) / (1-β₁ᵗ)  (바이어스 보정된 학습률)
///   p ← p - lr_t × m / (√v + ε)
///
/// 기본값: lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    ms: RefCell<Vec<ArrayD<f64>>>,
    vs: RefCell<Vec<ArrayD<f64>>>,
    t: Cell<u32>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            ms: RefCell::new(Vec::new()),
            vs: RefCell::new(Vec::new()),
            t: Cell::new(0),
        }
    }

    /// 파라미터를 Adam 규칙으로 업데이트
    /// params는 매 호출마다 동일한 순서로 전달해야 함 (모멘트 벡터와 1:1 대응)
    /// Lazy init: 첫 호출 시 또는 shape가 변하면 모멘트를 0으로 초기화
    pub fn update(&self, params: &[Variable]) {
        self.t.set(self.t.get() + 1);
        let t = self.t.get() as f64;
        let fix1 = 1.0 - self.beta1.powf(t);
        let fix2 = 1.0 - self.beta2.powf(t);
        let lr_t = self.lr * fix2.sqrt() / fix1;

        let mut ms = self.ms.borrow_mut();
        let mut vs = self.vs.borrow_mut();

        // 모멘트 벡터를 파라미터 수에 맞게 확장
        while ms.len() < params.len() {
            ms.push(ArrayD::zeros(ndarray::IxDyn(&[0])));
            vs.push(ArrayD::zeros(ndarray::IxDyn(&[0])));
        }

        for (i, p) in params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                // 첫 사용 시 올바른 shape로 초기화
                if ms[i].shape() != grad.shape() {
                    ms[i] = ArrayD::zeros(grad.raw_dim());
                    vs[i] = ArrayD::zeros(grad.raw_dim());
                }

                // m ← β₁·m + (1-β₁)·grad
                ms[i] = &ms[i] * self.beta1 + &grad * (1.0 - self.beta1);
                // v ← β₂·v + (1-β₂)·grad²
                vs[i] = &vs[i] * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);
                // p ← p - lr_t · m / (√v + ε)
                let update = ms[i].mapv(|m| m * lr_t) / vs[i].mapv(|v| v.sqrt() + self.eps);
                p.set_data(&p.data() - &update);
            }
        }
    }
}

/// AdamW 옵티마이저: Adam + 분리된 가중치 감쇠 (Weight Decay Decoupled)
///
/// Adam과의 차이:
///   Adam:  p ← p - lr_t × m / (√v + ε)                    ← L2 정규화와 혼합
///   AdamW: p ← p - lr_t × m / (√v + ε) - lr × wd × p     ← 가중치 감쇠를 분리
///
/// 왜 분리하는가?
///   Adam에서 L2 정규화(grad += wd * p)를 쓰면, 적응적 학습률 m/√v가
///   정규화 항도 스케일링해버린다 → 의도한 감쇠 강도와 달라짐
///   AdamW는 가중치 감쇠를 Adam 업데이트 밖에서 따로 적용 → 정확한 감쇠
///
/// Transformer 학습의 표준 옵티마이저 (GPT, BERT 모두 AdamW 사용)
pub struct AdamW {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    ms: RefCell<Vec<ArrayD<f64>>>,
    vs: RefCell<Vec<ArrayD<f64>>>,
    t: Cell<u32>,
}

impl AdamW {
    pub fn new(lr: f64, weight_decay: f64) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            ms: RefCell::new(Vec::new()),
            vs: RefCell::new(Vec::new()),
            t: Cell::new(0),
        }
    }

    pub fn update(&self, params: &[Variable]) {
        self.t.set(self.t.get() + 1);
        let t = self.t.get() as f64;
        let fix1 = 1.0 - self.beta1.powf(t);
        let fix2 = 1.0 - self.beta2.powf(t);
        let lr_t = self.lr * fix2.sqrt() / fix1;

        let mut ms = self.ms.borrow_mut();
        let mut vs = self.vs.borrow_mut();

        while ms.len() < params.len() {
            ms.push(ArrayD::zeros(ndarray::IxDyn(&[0])));
            vs.push(ArrayD::zeros(ndarray::IxDyn(&[0])));
        }

        for (i, p) in params.iter().enumerate() {
            if let Some(grad) = p.grad() {
                if ms[i].shape() != grad.shape() {
                    ms[i] = ArrayD::zeros(grad.raw_dim());
                    vs[i] = ArrayD::zeros(grad.raw_dim());
                }

                ms[i] = &ms[i] * self.beta1 + &grad * (1.0 - self.beta1);
                vs[i] = &vs[i] * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);
                let adam_update = ms[i].mapv(|m| m * lr_t)
                    / vs[i].mapv(|v| v.sqrt() + self.eps);
                // AdamW: 가중치 감쇠를 Adam 업데이트와 분리
                let wd_update = p.data().mapv(|w| w * self.lr * self.weight_decay);
                p.set_data(&(&p.data() - &adam_update) - &wd_update);
            }
        }
    }
}

// --- 데이터셋 ---

/// Dataset 트레잇: 데이터셋의 공통 인터페이스
/// Python의 dezero.datasets.Dataset 클래스에 해당
///
/// 왜 필요한가?
///   step48: get_spiral()이 (ArrayD, Vec<usize>) 튜플을 직접 반환
///           → 데이터셋마다 반환 형태가 다를 수 있고, 배치 추출 로직이 사용자 측에 노출
///   step49: Dataset 트레잇으로 통일된 인터페이스 제공
///           → len()과 get()만 있으면 어떤 데이터셋이든 동일한 학습 루프 사용 가능
pub trait Dataset {
    /// 데이터셋의 총 샘플 수
    fn len(&self) -> usize;

    /// i번째 샘플을 (입력 벡터, 라벨) 쌍으로 반환
    /// Python의 dataset[i]에 해당 (__getitem__)
    fn get(&self, index: usize) -> (Vec<f64>, usize);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Spiral 데이터셋 (3클래스 나선형 분류 문제)
/// Python의 dezero.datasets.Spiral 클래스에 해당
///
/// step48의 get_spiral()과 동일한 데이터를 생성하되,
/// Dataset 트레잇을 구현하여 개별 샘플 접근(get)과 크기 조회(len)를 지원
pub struct Spiral {
    data: ArrayD<f64>,   // (N, 2) 입력 좌표
    label: Vec<usize>,   // N개 클래스 라벨
}

impl Spiral {
    pub fn new(train: bool) -> Self {
        let (data, label) = get_spiral(train);
        Spiral { data, label }
    }
}

impl Dataset for Spiral {
    fn len(&self) -> usize {
        self.data.shape()[0]
    }

    fn get(&self, index: usize) -> (Vec<f64>, usize) {
        let cols = self.data.shape()[1];
        let x: Vec<f64> = (0..cols).map(|j| self.data[[index, j]]).collect();
        (x, self.label[index])
    }
}

/// Spiral 데이터셋의 원시 데이터 생성 (내부용)
/// 3개 클래스 × 100개 = 300개 샘플, 각 2D 좌표
pub fn get_spiral(train: bool) -> (ArrayD<f64>, Vec<usize>) {
    let seed: u64 = if train { 1984 } else { 2020 };
    let mut state: u64 = seed;

    let num_data = 100usize;
    let num_class = 3usize;
    let data_size = num_class * num_data;

    let mut x = vec![0.0f64; data_size * 2];
    let mut t = vec![0usize; data_size];

    for j in 0..num_class {
        for i in 0..num_data {
            let rate = i as f64 / num_data as f64;
            let radius = 1.0 * rate;

            // Box-Muller: 균일분포 → 정규분포 (노이즈 생성용)
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((state >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let noise =
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // 각 클래스의 나선 각도 = 기본 오프셋 + 진행 + 노이즈
            let theta = j as f64 * 4.0 + 4.0 * rate + noise * 0.2;
            let ix = num_data * j + i;
            x[ix * 2] = radius * theta.sin();
            x[ix * 2 + 1] = radius * theta.cos();
            t[ix] = j;
        }
    }

    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[data_size, 2]), x).unwrap();
    (x, t)
}

/// MNIST 손글씨 숫자 데이터셋
/// Python의 dezero.datasets.MNIST에 해당
///
/// MNIST (Modified National Institute of Standards and Technology):
///   1998년 Yann LeCun이 공개한 머신러닝의 "Hello World" 벤치마크.
///   손으로 쓴 숫자 0~9의 28×28 그레이스케일 이미지.
///
/// 데이터 흐름:
///   1) 첫 사용 시 인터넷에서 .gz 파일 4개를 다운로드 → ~/.dezero/mnist/에 캐시
///   2) flate2로 gzip 해제 → IDX 바이너리 형식 파싱
///   3) 각 28×28 이미지를 784차원 벡터로 flatten
///   4) 픽셀값 [0, 255] → [0, 1]로 정규화 (/255.0)
///      (정규화 이유: 원시 픽셀값이 크면 sigmoid 포화 → 기울기 소실 → 학습 실패)
///
/// 크기: train 60,000개, test 10,000개, 라벨 0~9 (10클래스)
pub struct MNIST {
    data: Vec<Vec<f64>>,  // N개 샘플, 각 784차원 (28×28 flatten, [0,1] 정규화)
    label: Vec<usize>,    // N개 라벨 (0~9 숫자)
}

impl MNIST {
    /// MNIST 데이터셋을 로드한다.
    /// train=true: 학습용 60,000개, train=false: 테스트용 10,000개
    ///
    /// 내부 흐름:
    ///   1) 캐시 디렉토리(~/.dezero/mnist/) 생성
    ///   2) 파일이 없으면 HTTP GET으로 다운로드 (ureq 사용)
    ///   3) .gz 파일을 flate2로 해제 → IDX 바이너리 파싱
    ///   4) 이미 캐시에 있으면 다운로드 스킵 → 즉시 파싱
    pub fn new(train: bool) -> Self {
        // PyTorch에서 사용하는 MNIST 미러 (원본 yann.lecun.com은 종종 불안정)
        let base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";

        // train/test에 따라 다른 파일 4개 중 2개를 선택
        //   train: train-images-idx3-ubyte.gz (~9.9MB), train-labels-idx1-ubyte.gz (~29KB)
        //   test:  t10k-images-idx3-ubyte.gz  (~1.6MB), t10k-labels-idx1-ubyte.gz  (~4.5KB)
        let (img_file, lbl_file) = if train {
            ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
        } else {
            ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
        };

        // 캐시 디렉토리: ~/.dezero/mnist/
        // Python DeZero도 동일하게 ~/.dezero에 캐시
        let cache_dir = dirs_cache_path();
        std::fs::create_dir_all(&cache_dir).expect("failed to create cache dir");

        let img_path = format!("{}/{}", cache_dir, img_file);
        let lbl_path = format!("{}/{}", cache_dir, lbl_file);

        // 다운로드: 캐시에 파일이 이미 있으면 스킵
        download_if_missing(&format!("{}{}", base_url, img_file), &img_path);
        download_if_missing(&format!("{}{}", base_url, lbl_file), &lbl_path);

        // IDX 바이너리 파일 파싱 → 메모리에 로드
        let data = load_mnist_images(&img_path);
        let label = load_mnist_labels(&lbl_path);

        MNIST { data, label }
    }
}

impl Dataset for MNIST {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> (Vec<f64>, usize) {
        (self.data[index].clone(), self.label[index])
    }
}

/// MNIST 캐시 디렉토리 경로를 반환 (~/.dezero/mnist/)
/// Python DeZero와 동일한 캐시 위치를 사용
fn dirs_cache_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{}/.dezero/mnist", home)
}

/// URL에서 파일 다운로드 (캐시에 없을 때만)
/// 파일이 이미 존재하면 즉시 반환 (캐시 히트)
/// ureq 크레이트로 동기 HTTP GET 요청 → 바이트 읽기 → 파일 저장
fn download_if_missing(url: &str, path: &str) {
    if std::path::Path::new(path).exists() {
        return; // 캐시 히트: 이미 다운로드된 파일 있음
    }
    println!("Downloading {} ...", url);
    let resp = ureq::get(url).call().expect("failed to download MNIST");
    let mut bytes = Vec::new();
    resp.into_body()
        .as_reader()
        .read_to_end(&mut bytes)
        .expect("failed to read response");
    std::fs::write(path, &bytes).expect("failed to write cache file");
    println!("Saved to {}", path);
}

/// gzip 압축된 IDX 이미지 파일을 파싱하여 Vec<Vec<f64>> 반환
///
/// IDX 파일 형식 (MNIST 자체 바이너리 포맷):
///   오프셋 0~3:   매직 넘버 0x00000803 (2051) — big-endian u32
///                 0x08 = unsigned byte, 0x03 = 3차원 (count × rows × cols)
///   오프셋 4~7:   이미지 수 (60000 또는 10000)
///   오프셋 8~11:  행 수 (28)
///   오프셋 12~15: 열 수 (28)
///   오프셋 16~:   픽셀 데이터 (각 이미지 784바이트, uint8)
///
/// 각 픽셀을 255.0으로 나누어 [0, 1] 범위로 정규화:
///   0 (검정) → 0.0, 255 (흰색) → 1.0
fn load_mnist_images(gz_path: &str) -> Vec<Vec<f64>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    // 1단계: gzip 해제 → 원본 바이너리를 메모리에 로드
    let file = std::fs::File::open(gz_path).expect("failed to open image file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).expect("failed to decompress");

    // 2단계: IDX 헤더 파싱 (모든 정수는 big-endian)
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2051, "invalid image magic number");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels = rows * cols; // 28 × 28 = 784

    // 3단계: 이미지 데이터 추출 (헤더 16바이트 이후부터)
    // 각 이미지: 연속된 784바이트 → uint8 → f64 / 255.0
    let data_start = 16;
    let mut images = Vec::with_capacity(count);
    for i in 0..count {
        let offset = data_start + i * pixels;
        let img: Vec<f64> = buf[offset..offset + pixels]
            .iter()
            .map(|&b| b as f64 / 255.0) // [0, 255] → [0.0, 1.0]
            .collect();
        images.push(img);
    }
    images
}

/// gzip 압축된 IDX 라벨 파일을 파싱하여 Vec<usize> 반환
///
/// IDX 라벨 파일 형식:
///   오프셋 0~3: 매직 넘버 0x00000801 (2049) — 0x01 = 1차원 (count만)
///   오프셋 4~7: 라벨 수 (60000 또는 10000)
///   오프셋 8~:  라벨 데이터 (각 1바이트, 값 0~9)
fn load_mnist_labels(gz_path: &str) -> Vec<usize> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = std::fs::File::open(gz_path).expect("failed to open label file");
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf).expect("failed to decompress");

    // 매직 넘버 2049 = 0x0801: unsigned byte 데이터, 1차원
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    assert_eq!(magic, 2049, "invalid label magic number");

    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;

    let data_start = 8;
    buf[data_start..data_start + count]
        .iter()
        .map(|&b| b as usize)
        .collect()
}

/// SinCurve 데이터셋: 사인 곡선의 다음 값을 예측하는 시계열 데이터
///
/// RNN 학습용: x[i] = sin(2π·i/T), t[i] = sin(2π·(i+1)/T)
/// 현재 값을 입력으로 받아 다음 시간 스텝의 값을 예측
///
/// train=true: 전체의 80% (앞부분)
/// train=false: 전체의 20% (뒷부분)
pub struct SinCurve {
    x: Vec<f64>,
    t: Vec<f64>,
}

impl SinCurve {
    pub fn new(train: bool) -> Self {
        let num_data = 1000;
        let period = 25.0;

        // 전체 사인 곡선 생성
        let y: Vec<f64> = (0..num_data)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin())
            .collect();

        let split = (num_data as f64 * 0.8) as usize; // 800

        let (x, t) = if train {
            // (y[0],y[1]), (y[1],y[2]), ..., (y[split-2],y[split-1])
            (y[..split - 1].to_vec(), y[1..split].to_vec())
        } else {
            (y[split - 1..num_data - 1].to_vec(), y[split..num_data].to_vec())
        };

        SinCurve { x, t }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// i번째 샘플: (입력값, 목표값)
    pub fn get(&self, i: usize) -> (f64, f64) {
        (self.x[i], self.t[i])
    }
}

/// SeqDataLoader: 시계열 데이터를 배치 단위로 순차 로딩
///
/// 일반 DataLoader와의 차이:
///   DataLoader: 랜덤 셔플 가능, 순서 무관 (분류 등)
///   SeqDataLoader: 시간 순서를 유지하면서 배치 병렬화 (RNN/LSTM 학습용)
///
/// 원리: 시퀀스를 batch_size개의 병렬 스트림으로 분할
///   데이터: [0, 1, 2, 3, 4, 5, 6, 7, 8], batch_size=3
///   스트림 0: [0, 1, 2]   스트림 1: [3, 4, 5]   스트림 2: [6, 7, 8]
///   t=0: batch = [0, 3, 6]
///   t=1: batch = [1, 4, 7]
///   t=2: batch = [2, 5, 8]
pub struct SeqDataLoader<'a> {
    dataset: &'a SinCurve,
    batch_size: usize,
    pub jump: usize,
    current: usize,
}

impl<'a> SeqDataLoader<'a> {
    pub fn new(dataset: &'a SinCurve, batch_size: usize) -> Self {
        let jump = dataset.len() / batch_size;
        SeqDataLoader {
            dataset,
            batch_size,
            jump,
            current: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }
}

impl<'a> Iterator for SeqDataLoader<'a> {
    type Item = (Variable, Variable);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.jump {
            return None;
        }

        let mut x_data = Vec::with_capacity(self.batch_size);
        let mut t_data = Vec::with_capacity(self.batch_size);

        for j in 0..self.batch_size {
            let idx = j * self.jump + self.current;
            let (x, t) = self.dataset.get(idx);
            x_data.push(x);
            t_data.push(t);
        }

        self.current += 1;

        let x = Variable::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[self.batch_size, 1]), x_data).unwrap(),
        );
        let t = Variable::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[self.batch_size, 1]), t_data).unwrap(),
        );

        Some((x, t))
    }
}

// --- DataLoader ---

/// DataLoader: Dataset을 배치 단위로 순회하는 이터레이터
/// Python의 dezero.DataLoader에 해당
///
/// step49까지는 셔플, 인덱스 슬라이싱, 배치 조립을 수동으로 처리했다.
/// DataLoader가 이 모든 것을 캡슐화:
///   for (x, t) in &mut loader { ... }
///
/// shuffle=true면 매 reset()마다 인덱스를 무작위로 섞음.
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    rng_state: u64,
    // 이터레이션 상태
    indices: Vec<usize>,
    current: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize, shuffle: bool) -> Self {
        let mut loader = DataLoader {
            dataset,
            batch_size,
            shuffle,
            rng_state: 0,
            indices: (0..dataset.len()).collect(),
            current: 0,
        };
        if shuffle {
            loader.shuffle_indices();
        }
        loader
    }

    /// 이터레이터를 처음으로 되돌림 (다음 에폭 시작)
    /// shuffle=true면 인덱스를 다시 섞음
    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            self.shuffle_indices();
        }
    }

    /// Fisher-Yates 셔플 (LCG 난수 사용)
    fn shuffle_indices(&mut self) {
        let n = self.indices.len();
        for i in (1..n).rev() {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((self.rng_state >> 11) as f64 / (1u64 << 53) as f64 * (i + 1) as f64)
                as usize;
            self.indices.swap(i, j);
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Variable, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let data_size = self.dataset.len();
        if self.current >= data_size {
            return None;
        }

        let start = self.current;
        let end = (start + self.batch_size).min(data_size);
        self.current = end;

        let batch_indices = &self.indices[start..end];
        let batch_len = end - start;

        // Dataset.get()으로 개별 샘플을 꺼내서 배치 조립
        let mut x_data = Vec::new();
        let mut t_data = Vec::with_capacity(batch_len);
        let mut input_dim = 0;

        for &idx in batch_indices {
            let (x, t) = self.dataset.get(idx);
            input_dim = x.len();
            x_data.extend_from_slice(&x);
            t_data.push(t);
        }

        let x = Variable::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[batch_len, input_dim]), x_data).unwrap(),
        );

        Some((x, t_data))
    }
}

// --- 벡터 검색 (Vector Search) ---

/// 벡터 유사도/거리 메트릭
/// Cosine, DotProduct는 유사도 (높을수록 유사)
/// L2, L1은 거리 (낮을수록 유사)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Cosine,
    DotProduct,
    L2,
    L1,
}

/// 코사인 유사도: cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
/// 범위 [-1, 1]. 방향만 비교하므로 스케일 불변.
pub fn cosine_similarity_vec(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b + 1e-10)
}

/// 내적 유사도: a · b = Σ aᵢbᵢ
/// 정규화된 벡터에서는 cosine과 동일. 크기 정보를 반영.
pub fn dot_product_vec(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 (유클리드) 거리: ‖a - b‖₂ = √Σ(aᵢ - bᵢ)²
/// 범위 [0, +∞). 가장 직관적인 "직선 거리".
pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| {
        let d = x - y;
        d * d
    }).sum::<f64>().sqrt()
}

/// L1 (맨해튼) 거리: ‖a - b‖₁ = Σ|aᵢ - bᵢ|
/// 범위 [0, +∞). 이상치에 L2보다 robust.
pub fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Brute Force 벡터 검색 인덱스
///
/// 모든 벡터를 메모리에 저장하고, 쿼리 시 전체를 선형 스캔하여 top-k를 반환.
/// 시간복잡도: O(N·D) per query. 정확한 최근접 탐색(exact NN)의 기준 구현.
/// 이후 IVF, PQ, HNSW의 recall을 검증하는 ground truth로 사용.
pub struct BruteForceIndex {
    vectors: Vec<Vec<f64>>,
    labels: Vec<String>,
    dim: usize,
}

impl BruteForceIndex {
    /// 빈 인덱스 생성. dim: 벡터 차원 (예: SentenceEmbedding의 n_embd)
    pub fn new(dim: usize) -> Self {
        BruteForceIndex {
            vectors: Vec::new(),
            labels: Vec::new(),
            dim,
        }
    }

    /// 단일 벡터 추가
    pub fn add(&mut self, vector: &[f64], label: &str) {
        debug_assert_eq!(vector.len(), self.dim, "vector dimension mismatch: expected {}, got {}", self.dim, vector.len());
        self.vectors.push(vector.to_vec());
        self.labels.push(label.to_string());
    }

    /// 배치 벡터 추가: (N, D) shape의 ArrayD에서 N개 벡터를 한 번에 추가
    pub fn add_batch(&mut self, vectors: &ArrayD<f64>, labels: &[String]) {
        let shape = vectors.shape();
        assert_eq!(shape.len(), 2, "expected 2D array, got {}D", shape.len());
        let n = shape[0];
        let d = shape[1];
        assert_eq!(d, self.dim, "dimension mismatch: expected {}, got {}", self.dim, d);
        assert_eq!(n, labels.len(), "vector count ({}) != label count ({})", n, labels.len());

        for i in 0..n {
            let row: Vec<f64> = (0..d).map(|j| vectors[[i, j]]).collect();
            self.vectors.push(row);
            self.labels.push(labels[i].clone());
        }
    }

    /// 쿼리 벡터에 대해 top-k 검색
    ///
    /// 반환: Vec<(인덱스, 점수, 라벨)>
    ///   - Cosine, DotProduct: 점수 내림차순 (높을수록 유사)
    ///   - L2, L1: 점수 오름차순 (낮을수록 유사)
    pub fn search(&self, query: &[f64], k: usize, metric: Metric) -> Vec<(usize, f64, String)> {
        debug_assert_eq!(query.len(), self.dim, "query dimension mismatch");

        if self.vectors.is_empty() {
            return Vec::new();
        }

        // 1. 모든 벡터에 대해 점수 계산
        let mut scores: Vec<(usize, f64)> = self.vectors.iter().enumerate().map(|(i, v)| {
            let score = match metric {
                Metric::Cosine => cosine_similarity_vec(query, v),
                Metric::DotProduct => dot_product_vec(query, v),
                Metric::L2 => l2_distance(query, v),
                Metric::L1 => l1_distance(query, v),
            };
            (i, score)
        }).collect();

        // 2. 정렬: 유사도는 내림차순, 거리는 오름차순
        match metric {
            Metric::Cosine | Metric::DotProduct => {
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            Metric::L2 | Metric::L1 => {
                scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // 3. top-k 반환
        let k = k.min(scores.len());
        scores[..k].iter().map(|&(i, score)| {
            (i, score, self.labels[i].clone())
        }).collect()
    }

    /// 배치 쿼리: 여러 쿼리를 한 번에 검색
    pub fn batch_search(&self, queries: &ArrayD<f64>, k: usize, metric: Metric) -> Vec<Vec<(usize, f64, String)>> {
        let shape = queries.shape();
        assert_eq!(shape.len(), 2, "expected 2D array");
        let d = shape[1];
        assert_eq!(d, self.dim, "dimension mismatch");

        let q = shape[0];
        (0..q).map(|i| {
            let row: Vec<f64> = (0..d).map(|j| queries[[i, j]]).collect();
            self.search(&row, k, metric)
        }).collect()
    }

    /// 저장된 벡터 수
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// 인덱스가 비어있는지 확인
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// i번째 벡터 반환
    pub fn get(&self, index: usize) -> &[f64] {
        &self.vectors[index]
    }

    /// 임베딩 차원
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// K-means 클러스터링 (Lloyd's 알고리즘)
///
/// 벡터 집합을 K개 클러스터로 분할. L2 거리 기반 할당 → 중심 갱신 반복.
/// IVF의 coarse quantizer로 사용: 공간을 K개 Voronoi cell로 분할.
///
/// 반환: (centroids [K][D], assignments [N])
pub fn kmeans(
    vectors: &[&[f64]],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let n = vectors.len();
    assert!(n >= k, "kmeans: need at least k={} vectors, got {}", k, n);
    let dim = vectors[0].len();

    // 1. Fisher-Yates로 k개의 고유 초기 중심 선택
    let mut rng = seed;
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = i + ((rng >> 11) as usize % (n - i));
        indices.swap(i, j);
    }
    let mut centroids: Vec<Vec<f64>> = (0..k).map(|i| vectors[indices[i]].to_vec()).collect();
    let mut assignments = vec![0usize; n];

    // 2. Lloyd's: assign → update → repeat
    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for c in 0..k {
                let dist = l2_distance(vectors[i], &centroids[c]);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }
        if !changed { break; }

        // Update
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                new_centroids[c][d] += vectors[i][d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_centroids[c][d] /= counts[c] as f64;
                }
            } else {
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }

    (centroids, assignments)
}

/// IVF (Inverted File Index) 근사 벡터 검색
///
/// 공간을 K개 클러스터(Voronoi cell)로 분할하고, 각 클러스터에 속한 벡터의
/// 역인덱스를 구축. 쿼리 시 nprobe개 가장 가까운 클러스터만 탐색.
/// O(N·D) → O(nprobe·N/K·D). nprobe=K이면 brute force와 동일.
pub struct IVFIndex {
    vectors: Vec<Vec<f64>>,
    labels: Vec<String>,
    dim: usize,
    n_clusters: usize,
    centroids: Vec<Vec<f64>>,
    inverted_lists: Vec<Vec<usize>>,
    is_trained: bool,
}

impl IVFIndex {
    /// 빈 IVF 인덱스 생성. 사용 전 train() 필요.
    pub fn new(dim: usize, n_clusters: usize) -> Self {
        IVFIndex {
            vectors: Vec::new(),
            labels: Vec::new(),
            dim,
            n_clusters,
            centroids: Vec::new(),
            inverted_lists: Vec::new(),
            is_trained: false,
        }
    }

    /// K-means로 클러스터 중심 학습
    pub fn train(&mut self, vectors: &[&[f64]], max_iter: usize, seed: u64) {
        assert!(!vectors.is_empty(), "train: need at least 1 vector");
        assert!(vectors.len() >= self.n_clusters,
            "train: need at least n_clusters={} vectors, got {}", self.n_clusters, vectors.len());

        let (centroids, _) = kmeans(vectors, self.n_clusters, max_iter, seed);
        self.centroids = centroids;
        self.inverted_lists = vec![Vec::new(); self.n_clusters];
        self.is_trained = true;
    }

    /// 단일 벡터 추가: 가장 가까운 중심의 역인덱스에 등록
    pub fn add(&mut self, vector: &[f64], label: &str) {
        assert!(self.is_trained, "IVFIndex: must call train() before add()");
        debug_assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let idx = self.vectors.len();
        self.vectors.push(vector.to_vec());
        self.labels.push(label.to_string());

        let cluster = self.nearest_centroid(vector);
        self.inverted_lists[cluster].push(idx);
    }

    /// 배치 벡터 추가
    pub fn add_batch(&mut self, vectors: &ArrayD<f64>, labels: &[String]) {
        assert!(self.is_trained, "IVFIndex: must call train() before add_batch()");
        let shape = vectors.shape();
        assert_eq!(shape.len(), 2, "expected 2D array, got {}D", shape.len());
        let n = shape[0];
        let d = shape[1];
        assert_eq!(d, self.dim, "dimension mismatch");
        assert_eq!(n, labels.len(), "vector count != label count");

        for i in 0..n {
            let row: Vec<f64> = (0..d).map(|j| vectors[[i, j]]).collect();
            self.add(&row, &labels[i]);
        }
    }

    /// 근사 최근접 탐색
    ///
    /// coarse search: nprobe개 가까운 centroid 찾기 (L2)
    /// fine search: 해당 클러스터 벡터만 user metric으로 정밀 검색
    pub fn search(&self, query: &[f64], k: usize, nprobe: usize, metric: Metric) -> Vec<(usize, f64, String)> {
        assert!(self.is_trained, "IVFIndex: must call train() before search()");
        debug_assert_eq!(query.len(), self.dim, "query dimension mismatch");

        if self.vectors.is_empty() {
            return Vec::new();
        }

        let nprobe = nprobe.min(self.n_clusters);

        // 1. Coarse: nprobe개 가장 가까운 중심 (L2)
        let mut centroid_dists: Vec<(usize, f64)> = self.centroids.iter().enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();
        centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. 후보 수집
        let mut candidates: Vec<usize> = Vec::new();
        for &(cluster_id, _) in centroid_dists.iter().take(nprobe) {
            candidates.extend_from_slice(&self.inverted_lists[cluster_id]);
        }

        // 3. Fine search
        let mut scores: Vec<(usize, f64)> = candidates.iter().map(|&i| {
            let score = match metric {
                Metric::Cosine => cosine_similarity_vec(query, &self.vectors[i]),
                Metric::DotProduct => dot_product_vec(query, &self.vectors[i]),
                Metric::L2 => l2_distance(query, &self.vectors[i]),
                Metric::L1 => l1_distance(query, &self.vectors[i]),
            };
            (i, score)
        }).collect();

        // 4. 정렬
        match metric {
            Metric::Cosine | Metric::DotProduct => {
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            Metric::L2 | Metric::L1 => {
                scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        let k = k.min(scores.len());
        scores[..k].iter().map(|&(i, score)| {
            (i, score, self.labels[i].clone())
        }).collect()
    }

    /// 배치 쿼리
    pub fn batch_search(&self, queries: &ArrayD<f64>, k: usize, nprobe: usize, metric: Metric) -> Vec<Vec<(usize, f64, String)>> {
        let shape = queries.shape();
        assert_eq!(shape.len(), 2, "expected 2D array");
        assert_eq!(shape[1], self.dim, "dimension mismatch");

        let q = shape[0];
        (0..q).map(|i| {
            let row: Vec<f64> = (0..shape[1]).map(|j| queries[[i, j]]).collect();
            self.search(&row, k, nprobe, metric)
        }).collect()
    }

    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn is_trained(&self) -> bool { self.is_trained }

    /// 각 클러스터의 벡터 수 (디버깅용)
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.inverted_lists.iter().map(|list| list.len()).collect()
    }

    fn nearest_centroid(&self, vector: &[f64]) -> usize {
        let mut best = 0;
        let mut best_dist = f64::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let dist = l2_distance(vector, c);
            if dist < best_dist {
                best_dist = dist;
                best = i;
            }
        }
        best
    }
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
