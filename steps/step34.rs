// step34: sin(x)의 고차 미분 (이중 역전파 반복)
// backward(create_graph=true)를 반복하여 sin의 1~3차 도함수를 자동 계산
// sin → cos → -sin → -cos

use dezero::{sin, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_higher_order_derivatives_of_sin() {
        // -7.0부터 7.0까지 균등하게 200개의 숫자를 생성
        let x_data = ndarray::Array1::linspace(-7.0, 7.0, 200).into_dyn();
        let x = Variable::new(x_data.clone());

        // 시작은 sin(x)
        let y = sin(&x);
        y.backward(false, true);

        // logs[0] = sin(x), 이후 루프에서 1~3차 도함수 추가
        // 테스트 검사용 과정 기록
        let mut logs = vec![y.data()];

        for _ in 0..3 {
            logs.push(x.grad().unwrap());
            let gx = x.grad_var().unwrap();
            x.cleargrad();
            gx.backward(false, true);
        }

        // logs = [sin(x), cos(x), -sin(x), -cos(x)]
        let expected = [
            x_data.mapv(f64::sin),
            x_data.mapv(f64::cos),
            x_data.mapv(|v| -v.sin()),
            x_data.mapv(|v| -v.cos()),
        ];
        let labels = ["sin(x)", "cos(x)", "-sin(x)", "-cos(x)"];

        for (i, (log, exp)) in logs.iter().zip(expected.iter()).enumerate() {
            let max_err = (log - exp)
                .mapv(f64::abs)
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max);
            println!("{}: max error = {:.2e}", labels[i], max_err);
            assert!(max_err < 1e-10, "{} error too large: {}", labels[i], max_err);
        }
    }
}
