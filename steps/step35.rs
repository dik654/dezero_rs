// step35: tanh(x)의 고차 미분과 계산 그래프 시각화
// backward(create_graph=true)를 반복하여 tanh의 고차 도함수를 자동 계산하고
// 계산 그래프를 DOT/Graphviz로 출력

use dezero::{plot_dot_graph, tanh, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_higher_order_derivatives() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = tanh(&x);
        y.backward(false, true);

        let iters = 1;

        for _ in 0..iters {
            let gx = x.grad_var().unwrap();
            x.cleargrad();
            gx.backward(false, true);
        }

        let gx = x.grad_var().unwrap();
        gx.set_name(&format!("gx{}", iters + 1));
        let _ = plot_dot_graph(&gx, false, "tanh.png");

        // tanh(1.0) 값 검증
        let tanh_val = 1.0_f64.tanh();
        let y_val = *y.data().iter().next().unwrap();
        assert!(
            (y_val - tanh_val).abs() < 1e-10,
            "tanh(1.0) should be {}, got {}",
            tanh_val,
            y_val
        );

        // tanh'(x) = 1 - tanh(x)^2
        // tanh'(1.0) = 1 - tanh(1.0)^2
        let expected_grad1 = 1.0 - tanh_val * tanh_val;
        // iters=1이면 2차 미분: tanh''(x) = -2*tanh(x)*(1-tanh(x)^2)
        let expected_grad2 = -2.0 * tanh_val * (1.0 - tanh_val * tanh_val);
        let gx_val = *gx.data().iter().next().unwrap();
        assert!(
            (gx_val - expected_grad2).abs() < 1e-10,
            "tanh''(1.0) should be {}, got {}",
            expected_grad2,
            gx_val
        );

        println!("tanh(1.0) = {:.10}", y_val);
        println!("tanh'(1.0) = {:.10}", expected_grad1);
        println!("tanh''(1.0) = {:.10}", gx_val);
    }
}
