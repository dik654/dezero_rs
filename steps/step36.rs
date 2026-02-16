// step36: 이중 역전파로 얻은 기울기를 새 계산에 활용
// y = x^2의 기울기 gx(=2x)를 Variable로 유지한 뒤
// z = gx^3 + y 를 구성하고 z.backward()로 dz/dx를 자동 계산
// z = (2x)^3 + x^2 = 8x^3 + x^2 → dz/dx = 24x^2 + 2x

use dezero::Variable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_in_new_computation() {
        let x = Variable::new(ndarray::arr0(2.0).into_dyn());

        // y = x^2
        let y = x.pow(2.0);
        // create_graph=true → gx가 Variable로 유지 (creator 체인 보존)
        y.backward(false, true);

        // gx = dy/dx = 2x (단순 숫자가 아닌 Variable)
        let gx = x.grad_var().unwrap();
        x.cleargrad();

        // gx를 새 수식의 재료로 활용: z = gx^3 + y
        // gx = 2x이므로 z = (2x)^3 + x^2 = 8x^3 + x^2
        let z = &gx.pow(3.0) + &y;
        z.backward(false, false);

        // dz/dx = 24x^2 + 2x = 24*4 + 4 = 100
        let result = *x.grad().unwrap().iter().next().unwrap();
        println!("x.grad = {}", result);
        assert!(
            (result - 100.0).abs() < 1e-10,
            "dz/dx at x=2 should be 100, got {}",
            result
        );
    }
}
