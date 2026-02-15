// step19: Variable 사용성 개선
// name 필드로 변수에 이름을 붙이고
// shape, ndim, size, len 등 형상 정보 접근 메서드와
// Display 트레잇으로 출력 포맷을 추가한다

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
    name: Option<String>, // 변수 이름 (디버깅/시각화용)
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

    /// 이름을 붙여서 생성 (Python의 Variable(data, name=...))
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

    // --- 형상 정보 접근 메서드 (Python의 @property에 해당) ---
    // 형상이란 배열의 모양(크기) 정보로 텐서를 다룰 때 자주 확인

    /// 배열의 형상 (예: [2, 3])
    fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    /// 차원 수 (스칼라=0, 벡터=1, 행렬=2, ...)
    fn ndim(&self) -> usize {
        self.inner.borrow().data.ndim()
    }

    /// 전체 원소 수
    fn size(&self) -> usize {
        self.inner.borrow().data.len()
    }

    /// 첫 번째 축의 길이 (Python의 __len__)
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

/// Python의 __repr__ 에 해당
/// println!("{}", x) 로 출력 가능
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
    use ndarray::{arr0, arr1, arr2};

    #[test]
    fn test_name() {
        let x = Variable::with_name(arr0(1.0).into_dyn(), "x");
        assert_eq!(
            x.inner.borrow().name.as_deref(),
            Some("x")
        );

        // 이름 없이 생성
        let y = Variable::new(arr0(1.0).into_dyn());
        assert!(y.inner.borrow().name.is_none());
    }

    #[test]
    fn test_shape_scalar() {
        let x = Variable::new(arr0(1.0).into_dyn());
        assert_eq!(x.shape(), Vec::<usize>::new()); // 스칼라: 빈 shape
        assert_eq!(x.ndim(), 0);
        assert_eq!(x.size(), 1);
        assert_eq!(x.len(), 0); // 스칼라는 축이 없으므로 0
    }

    #[test]
    fn test_shape_vector() {
        let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(x.shape(), vec![3]);
        assert_eq!(x.ndim(), 1);
        assert_eq!(x.size(), 3);
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn test_shape_matrix() {
        let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
        assert_eq!(x.shape(), vec![2, 3]);
        assert_eq!(x.ndim(), 2);
        assert_eq!(x.size(), 6);
        assert_eq!(x.len(), 2); // 첫 번째 축의 길이
    }

    #[test]
    fn test_display() {
        let x = Variable::new(arr0(3.0).into_dyn());
        let s = format!("{}", x);
        assert!(s.starts_with("variable("));

        let x = Variable::with_name(arr0(3.0).into_dyn(), "x");
        let s = format!("{}", x);
        assert!(s.contains("name=x"));
    }
}
