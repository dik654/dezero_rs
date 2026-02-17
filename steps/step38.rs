// step38: reshape와 transpose 함수의 순전파/역전파
// reshape: shape 변환 후 역전파에서 원래 shape 복원
// transpose: 전치 후 역전파에서 다시 전치

use dezero::{reshape, transpose, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_backward() {
        // (2,3) -> (6,) reshape 후 역전파에서 (2,3)으로 복원되는지 확인
        let x = Variable::new(ndarray::array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]].into_dyn());
        let y = reshape(&x, &[6]);
        y.backward(true, false);

        let x_grad = x.grad().unwrap();
        println!("x.grad = {}", x_grad);
        // reshape의 역전파: 기울기 shape가 원래 (2,3)으로 복원
        assert_eq!(x_grad.shape(), &[2, 3]);
        assert_eq!(x_grad, ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());
    }

    #[test]
    fn test_transpose_backward() {
        // (2,3) -> (3,2) transpose 후 역전파에서 (2,3)으로 복원되는지 확인
        let x = Variable::new(ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let y = transpose(&x);
        y.backward(false, false);

        let x_grad = x.grad().unwrap();
        println!("x.grad = {}", x_grad);
        // transpose의 역전파: 기울기 shape가 원래 (2,3)으로 복원
        assert_eq!(x_grad.shape(), &[2, 3]);
        assert_eq!(x_grad, ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());
    }
}
