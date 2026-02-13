use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// Function의 상태를 담는 구조체
/// input 저장을 각 구현체(Square, Exp)가 아닌 여기서 공통으로 관리한다
struct FuncState {
    // Box를 사용하여 힙에 할당하고 포인터(고정 크기)만 저장
    func: Box<dyn Function>,
    input: Option<Variable>,
}

type FuncStateRef = Rc<RefCell<FuncState>>;

struct VarInner {
    data: ArrayD<f64>,
    grad: Option<ArrayD<f64>>,
    creator: Option<FuncStateRef>,
}

/// Variable의 newtype 래퍼
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
        // 한 곳에서만 쓰기 가능(borrow_mut)
        self.inner.borrow_mut().creator = Some(Rc::clone(state));
    }

    fn backward(&self) {
        // 이 변수를 만든 함수(creator)를 가져온다
        let creator = self.inner.borrow().creator.clone();
        // creator가 있으면 역전파 진행 (없으면 입력 변수이므로 종료)
        if let Some(state_ref) = creator {
            // 현재 변수의 기울기(grad) 가져오기 (출력쪽에서 전해진 미분값)
            let grad = self.inner.borrow().grad.clone().unwrap();
            let (gx, input) = {
                let state = state_ref.borrow();
                // creator 함수의 입력 변수를 가져오기
                let input = state.input.clone().unwrap();
                // 입력 변수의 데이터(x)를 가져오기
                let x = input.inner.borrow().data.clone();
                // ** 역전파 자동화의 핵심 ** creator 함수의 backward를 호출하여 입력쪽 기울기(gx)를 계산
                let gx = state.func.backward(&x, &grad);
                (gx, input)
            };
            // 계산한 기울기를 입력 변수의 grad에 저장
            input.inner.borrow_mut().grad = Some(gx);
            // 입력 변수의 backward를 재귀 호출하여 더 앞쪽으로 역전파를 잇기
            input.backward();
        }
    }
}

/// Function trait: forward와 backward만 구현하면 된다
/// input 저장은 FuncState가 공통으로 처리하므로 구현체에서 신경 쓸 필요 없다
trait Function {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64>;
    /// backward에 x(입력 데이터)가 인수로 전달되므로 self에 저장할 필요 없다
    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64>;
}

/// Function의 newtype 래퍼
struct Func {
    state: FuncStateRef,
}

impl Func {
    fn new(func: impl Function + 'static) -> Self {
        Func {
            state: Rc::new(RefCell::new(FuncState {
                func: Box::new(func),
                input: None,
            })),
        }
    }

    /// Python의 __call__에 해당
    fn call(&self, input: &Variable) -> Variable {
        let x_data = input.inner.borrow().data.clone();
        let y_data = {
            let mut state = self.state.borrow_mut();
            state.input = Some(input.clone());
            state.func.forward(&x_data)
        };
        let output = Variable::new(y_data);
        output.set_creator(&self.state);
        output
    }
}

/// 제곱 연산: forward와 backward만 구현
struct Square;

impl Square {
    fn new() -> Func {
        Func::new(Square)
    }
}

impl Function for Square {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        x * x
    }

    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        2.0 * x * gy
    }
}

/// 지수 연산: forward와 backward만 구현
struct Exp;

impl Exp {
    fn new() -> Func {
        Func::new(Exp)
    }
}

impl Function for Exp {
    fn forward(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        x.mapv(f64::exp)
    }

    fn backward(&self, x: &ArrayD<f64>, gy: &ArrayD<f64>) -> ArrayD<f64> {
        x.mapv(f64::exp) * gy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_backward() {
        let a = Square::new();
        let b = Exp::new();
        let c = Square::new();

        // 순전파: x -> Square -> Exp -> Square -> y
        let x = Variable::new(ndarray::arr0(0.5).into_dyn());
        let a_out = a.call(&x);
        let b_out = b.call(&a_out);
        let y = c.call(&b_out);

        // y.backward() 한 번으로 자동 역전파
        y.inner.borrow_mut().grad = Some(ndarray::arr0(1.0).into_dyn());
        y.backward();

        let expected = 4.0 * 0.5 * (2.0 * 0.5_f64.powi(2)).exp();
        let result = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert!((result - expected).abs() < 1e-10);
    }
}
