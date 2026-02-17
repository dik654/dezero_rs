// step37: sum 함수의 역전파에서 shape 변환이 올바른지 검증
// 순전파: (2,3) 행렬 → 스칼라로 축소
// 역전파: 스칼라 → (2,3) 행렬로 복원
// 모든 연산이 덧셈뿐이라 기울기 값은 전부 1이지만
// 핵심은 값이 아니라 shape가 제대로 복원되는지 확인하는 것

use dezero::{sum, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_backward() {
        // x = [[1,2,3],[4,5,6]], c = [[10,20,30],[40,50,60]]
        let x = Variable::new(ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let c = Variable::new(ndarray::array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn());

        // t = x + c, y = sum(t)
        let t = &x + &c;
        let y = sum(&t);

        // retain_grad=true로 중간 변수 t의 기울기도 보존
        y.backward(true, false);

        // y.grad = 1 (스칼라) → sum 결과가 스칼라인지 확인
        let y_grad = y.grad().unwrap();
        println!("y.grad = {}", y_grad);
        assert_eq!(*y_grad.iter().next().unwrap(), 1.0);

        // t.grad = ones(2,3) → 스칼라에서 (2,3)으로 shape 복원 확인
        let t_grad = t.grad().unwrap();
        println!("t.grad = {}", t_grad);
        assert_eq!(t_grad, ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());

        // x.grad = ones(2,3) → Add 역전파가 같은 shape 유지 확인
        let x_grad = x.grad().unwrap();
        println!("x.grad = {}", x_grad);
        assert_eq!(x_grad, ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());

        // c.grad = ones(2,3) → Add의 다른 입력도 같은 shape 유지 확인
        let c_grad = c.grad().unwrap();
        println!("c.grad = {}", c_grad);
        assert_eq!(c_grad, ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());
    }
}
