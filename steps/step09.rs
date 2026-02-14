use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

/// Function의 상태를 담는 구조체
/// input 저장을 각 구현체(Square, Exp)를 여기서 공통으로 관리
struct FuncState {
    func: Box<dyn Function>,
    input: Option<Variable>,
    output: Option<Variable>,
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
        self.inner.borrow_mut().creator = Some(Rc::clone(state));
    }

    /// step07에서는 재귀였지만, step09에서는 반복문으로 변경
    /// 재귀는 계산 그래프가 깊어지면 스택 오버플로우 위험이 있음
    fn backward(&self) {
        {
            let mut inner = self.inner.borrow_mut();
            // grad가 None이면 1.0으로 초기화 (Python의 if self.grad is None)
            if inner.grad.is_none() {
                inner.grad = Some(ArrayD::ones(inner.data.shape()));
            }
        }

        // 처리할 함수들을 담는 리스트 (Python의 funcs = [self.creator])
        let mut funcs: Vec<FuncStateRef> = Vec::new();
        if let Some(creator) = self.inner.borrow().creator.clone() {
            funcs.push(creator);
        }

        // 리스트가 빌 때까지 반복 (Python의 while funcs:)
        while let Some(state_ref) = funcs.pop() {
            let (gx, input) = {
                let state = state_ref.borrow();
                // f.output에서 기울기를 가져온다
                let output = state.output.as_ref().unwrap();
                let gy = output.inner.borrow().grad.clone().unwrap();
                // f.input에서 입력 변수와 데이터를 가져온다
                // input은 블록 밖에서도 사용하기에 clone
                let input = state.input.clone().unwrap();
                // x는 내부에서만 사용하므로 빌려서 값만 읽기
                let x = input.inner.borrow().data.clone();
                // 각 내부 구현되어있는 기울기 계산 실행 (Square, Exp 등)
                let gx = state.func.backward(&x, &gy);
                (gx, input)
            };
            // 최종 기울기는 입력 변수의 기울기
            input.inner.borrow_mut().grad = Some(gx);
            // 입력 변수에 creator가 있으면 리스트에 추가하여 역전파를 이어간다
            if let Some(creator) = input.inner.borrow().creator.clone() {
                funcs.push(creator);
            }
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
    // impl Function: Function trait을 구현한 타입이면 무엇이든 받음 (Square, Exp 등)
    // 'static: 내부에 빌린 참조가 없어야 Box에 안전하게 저장 가능
    fn new(func: impl Function + 'static) -> Self {
        Func {
            // Rc: 여러 곳에서 공유 가능하게
            // RefCell: 런타임에 읽기/쓰기 전환 가능하게
            // FuncState: 실제 데이터
            state: Rc::new(RefCell::new(FuncState {
                func: Box::new(func),
                input: None,
                output: None,
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
        // 출력 변수를 FuncState에 저장
        self.state.borrow_mut().output = Some(output.clone());
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

/// Python의 square(x) 편의 함수
/// Square()(x) 처럼 매번 인스턴스를 만들고 호출하는 것을 간결하게 한다
fn square(x: &Variable) -> Variable {
    Square::new().call(x)
}

/// Python의 exp(x) 편의 함수
fn exp(x: &Variable) -> Variable {
    Exp::new().call(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_backward() {
        // 편의 함수로 간결하게 순전파: square(exp(square(x)))
        let x = Variable::new(ndarray::arr0(0.5).into_dyn());
        let y = square(&exp(&square(&x)));

        // grad를 직접 설정하지 않아도 자동으로 1.0으로 초기화된다
        y.backward();

        let expected = 4.0 * 0.5 * (2.0 * 0.5_f64.powi(2)).exp();
        let result = *x.inner.borrow().grad.as_ref().unwrap().first().unwrap();
        assert!((result - expected).abs() < 1e-10);
    }
}
