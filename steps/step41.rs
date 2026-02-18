// step41: 행렬곱(matmul)의 순전파/역전파
// 순전파: y = x @ w
// 역전파: gx = gy @ w^T, gw = x^T @ gy

use dezero::{matmul, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_backward() {
        // x (2,3) @ w (3,4) = y (2,4)
        let x = Variable::new(ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap());
        let w = Variable::new(ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap());

        // 형렬곱 실행
        let y = matmul(&x, &w);
        y.backward(false, false);

        // 기울기의 shape도 행렬의 shape와 동일
        // x.grad shape = x shape (2,3)
        // w.grad shape = w shape (3,4)
        println!("x.grad.shape = {:?}", x.grad().unwrap().shape());
        println!("w.grad.shape = {:?}", w.grad().unwrap().shape());
        assert_eq!(x.grad().unwrap().shape(), &[2, 3]);
        assert_eq!(w.grad().unwrap().shape(), &[3, 4]);
    }
}
