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

/// Python의 using_config('enable_backprop', value)에 해당
/// 역전파 그래프 생성을 enable 값으로 설정하고, 스코프 종료 시 이전 값 복원
fn using_backprop(enable: bool) -> NoGradGuard {
    let prev = ENABLE_BACKPROP.with(|c| c.get());
    ENABLE_BACKPROP.with(|c| c.set(enable));
    NoGradGuard { prev }
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
        // 기울기를 원래 입력의 shape로 브로드캐스트
        let gy = &gys[0];
        // 기울기의 shape를 입력 shape에 맞게 브로드캐스트 가능한 형태로 변환
        let gy_data = gy.data();
        let broadcast = gy_data.broadcast(ndarray::IxDyn(&self.x_shape)).unwrap().to_owned();
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

pub fn matmul(x: &Variable, w: &Variable) -> Variable {
    Func::new(MatMulFn).call(&[x, w])
}

pub fn sigmoid(x: &Variable) -> Variable {
    Func::new(SigmoidFn).call(&[x])
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
