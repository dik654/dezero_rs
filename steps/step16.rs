use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

struct FuncState {
    func: Box<dyn Function>,
    // 세대 수가 큰 쪽부터 처리하면 부모보다 자식이 먼저 처리됨 보장
    generation: u32, // 세대: 입력 변수들의 최대 세대
    inputs: Vec<Variable>,
    outputs: Vec<Variable>,
}

type FuncStateRef = Rc<RefCell<FuncState>>;

struct VarInner {
    data: ArrayD<f64>,
    grad: Option<ArrayD<f64>>,
    creator: Option<FuncStateRef>,
    generation: u32, // 세대: creator의 세대 + 1 (입력 변수는 0)
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
        // 출력 변수의 세대 = 함수의 세대 + 1
        inner.generation = func_gen + 1;
    }

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
        // 중복 방지: 같은 함수를 두 번 처리하지 않는다
        // Rc의 포인터 주소로 동일성 판별
        let mut seen: HashSet<*const RefCell<FuncState>> = HashSet::new();

        // 함수를 세대 순으로 정렬하며 추가하는 헬퍼
        let add_func = |f: FuncStateRef, funcs: &mut Vec<FuncStateRef>, seen: &mut HashSet<*const RefCell<FuncState>>| {
            let ptr = Rc::as_ptr(&f);
            // 이 주소를 봤는지 확인하고 안 봤다면
            if !seen.contains(&ptr) {
                // 이 주소를 본 것으로 기록
                seen.insert(ptr);
                // 앞으로 처리할 함수 대기열 끝에 추가
                funcs.push(f);
                // 세대 오름차순 정렬 → pop()하면 가장 높은 세대가 나온다
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

        // 함수의 세대 = 입력 변수들의 최대 세대
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
        self.state.borrow_mut().outputs = outputs.clone();
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
        // dy/dx = 2x
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

    /// 복잡한 계산 그래프: y = (a²) + (a²), a = x²
    /// 세대 순서가 없으면 틀린 결과가 나온다
    #[test]
    fn test_complex_graph() {
        // x=2, a=x²=4, y=a²+a²=32
        // dy/da = 2a + 2a = 4a = 16
        // da/dx = 2x = 4
        // dy/dx = 16 * 4 = 64
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward();

        let x_grad = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert_eq!(x_grad, 64.0);
    }

    /// cleargrad 후 다시 계산
    #[test]
    fn test_complex_graph_cleargrad() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());

        // 1회차
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward();
        assert_eq!(
            *x.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            64.0
        );

        // cleargrad 후 2회차
        x.cleargrad();
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward();
        assert_eq!(
            *x.inner.borrow().grad.as_ref().unwrap().first().unwrap(),
            64.0
        );
    }
}
