// step39: sum 함수에 axis, keepdims 옵션 추가
// axis: 특정 축 방향으로만 합산 (None이면 전체 합산)
// keepdims: 합산 후에도 차원 수를 유지할지 여부

use dezero::{sum, sum_with, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_1d() {
        // 1차원 배열 전체 합산
        let x = Variable::new(ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_dyn());
        let y = sum(&x);
        y.backward(false, false);

        println!("y = {}", y.data());
        println!("x.grad = {}", x.grad().unwrap());
        assert_eq!(*y.data().iter().next().unwrap(), 21.0);
        assert_eq!(x.grad().unwrap(), ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0].into_dyn());
    }

    #[test]
    fn test_sum_2d() {
        // 2차원 배열 전체 합산
        let x = Variable::new(ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        let y = sum(&x);
        y.backward(false, false);

        println!("y = {}", y.data());
        println!("x.grad = {}", x.grad().unwrap());
        assert_eq!(*y.data().iter().next().unwrap(), 21.0);
        assert_eq!(x.grad().unwrap(), ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());
    }

    #[test]
    fn test_sum_axis0() {
        // axis=0: 행 방향 합산 (2,3) -> (3,)
        // [[1,2,3],[4,5,6]] → [5,7,9]
        let x = Variable::new(ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn());
        // 옵션 지정 가능
        // axis=Some(0): [[1,2,3],[4,5,6]] → [5,7,9] (축 방향 합산)
        // keepdims=true: 합산 후 차원 수 유지 (2,3,4,5) → (1,1,1,1)
        let y = sum_with(&x, Some(0), false);
        y.backward(false, false);

        println!("y = {}", y.data());
        println!("x.grad = {}", x.grad().unwrap());
        assert_eq!(y.data(), ndarray::array![5.0, 7.0, 9.0].into_dyn());
        assert_eq!(x.grad().unwrap(), ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn());
    }

    #[test]
    fn test_sum_keepdims() {
        // keepdims=true: 합산 후 차원 수 유지
        // shape (2,3,4,5) -> (1,1,1,1)
        let x = Variable::new(ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[2, 3, 4, 5])));
        // axis=None: 전체 합산 → 스칼라 (기존과 동일)
        // keepdims=true: 합산 후 차원 수 유지 (2,3,4,5) → (1,1,1,1)
        let y = sum_with(&x, None, true);
        println!("y.shape = {:?}", y.shape());
        assert_eq!(y.shape(), vec![1, 1, 1, 1]);
    }
}
