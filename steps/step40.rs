// step40: 브로드캐스트 역전파
// forward에서 shape가 다른 배열끼리 연산하면 ndarray가 자동 브로드캐스트
// backward에서는 기울기를 원래 shape로 축소해야 함

use dezero::Variable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_add() {
        // x0 shape (3,) + x1 shape (1,) → y shape (3,)
        // backward: x1.grad는 shape (3,) 기울기를 shape (1,)로 합산
        let x0 = Variable::new(ndarray::array![1.0, 2.0, 3.0].into_dyn());
        let x1 = Variable::new(ndarray::array![10.0].into_dyn());
        let y = &x0 + &x1;

        println!("y = {}", y.data());
        assert_eq!(y.data(), ndarray::array![11.0, 12.0, 13.0].into_dyn());

        y.backward(false, false);

        println!("x1.grad = {}", x1.grad().unwrap());
        // x1은 3개 원소에 각각 더해졌으므로 기울기 = 1+1+1 = 3
        assert_eq!(x1.grad().unwrap(), ndarray::array![3.0].into_dyn());
        assert_eq!(x0.grad().unwrap(), ndarray::array![1.0, 1.0, 1.0].into_dyn());
    }
}
