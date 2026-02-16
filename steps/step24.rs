// step24: 복잡한 함수의 미분
// 라이브러리를 사용해 Sphere, Matyas, Goldstein-Price 함수의 편미분을 검증
// 연산자 오버로딩 덕분에 수식을 거의 그대로 코드로 옮길 수 있다

use dezero::Variable;

/// z = x^2 + y^2
fn sphere(x: &Variable, y: &Variable) -> Variable {
    &x.pow(2.0) + &y.pow(2.0)
}

/// z = 0.26(x^2 + y^2) - 0.48xy
fn matyas(x: &Variable, y: &Variable) -> Variable {
    let x2_y2 = &x.pow(2.0) + &y.pow(2.0);
    let xy = x * y;
    let term1 = 0.26 * &x2_y2;
    let term2 = 0.48 * &xy;
    &term1 - &term2
}

/// Goldstein-Price function
/// z = (1 + (x+y+1)^2 * (19 - 14x + 3x^2 - 14y + 6xy + 3y^2))
///   * (30 + (2x-3y)^2 * (18 - 32x + 12x^2 + 48y - 36xy + 27y^2))
fn goldstein(x: &Variable, y: &Variable) -> Variable {
    // a = x + y + 1
    let sum_xy = x + y;
    let a = &sum_xy + 1.0;

    // b = 19 - 14x + 3x^2 - 14y + 6xy + 3y^2
    let b1 = 19.0 - &(14.0 * x);
    let b2 = &b1 + &(3.0 * &x.pow(2.0));
    let b3 = &b2 - &(14.0 * y);
    let b4 = &b3 + &(6.0 * &(x * y));
    let b = &b4 + &(3.0 * &y.pow(2.0));

    // part1 = 1 + a^2 * b
    let a2b = &a.pow(2.0) * &b;
    let part1 = 1.0 + &a2b;

    // c = 2x - 3y
    let c = &(2.0 * x) - &(3.0 * y);

    // d = 18 - 32x + 12x^2 + 48y - 36xy + 27y^2
    let d1 = 18.0 - &(32.0 * x);
    let d2 = &d1 + &(12.0 * &x.pow(2.0));
    let d3 = &d2 + &(48.0 * y);
    let d4 = &d3 - &(36.0 * &(x * y));
    let d = &d4 + &(27.0 * &y.pow(2.0));

    // part2 = 30 + c^2 * d
    let c2d = &c.pow(2.0) * &d;
    let part2 = 30.0 + &c2d;

    &part1 * &part2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_grad(v: &Variable) -> f64 {
        *v.grad().unwrap().first().unwrap()
    }

    /// sphere(1, 1) = 2, ∂z/∂x = 2x = 2, ∂z/∂y = 2y = 2
    #[test]
    fn test_sphere() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = Variable::new(ndarray::arr0(1.0).into_dyn());
        let z = sphere(&x, &y);
        z.backward(false);

        assert_eq!(get_grad(&x), 2.0);
        assert_eq!(get_grad(&y), 2.0);
    }

    /// matyas(1, 2):
    /// ∂z/∂x = 0.52x - 0.48y = 0.52 - 0.96 = -0.44
    /// ∂z/∂y = 0.52y - 0.48x = 1.04 - 0.48 = 0.56
    #[test]
    fn test_matyas() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = Variable::new(ndarray::arr0(2.0).into_dyn());
        let z = matyas(&x, &y);
        z.backward(false);

        assert!((get_grad(&x) - (-0.44)).abs() < 1e-10);
        assert!((get_grad(&y) - 0.56).abs() < 1e-10);
    }

    /// goldstein(1, 1): Python 결과와 동일한지 검증
    #[test]
    fn test_goldstein() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = Variable::new(ndarray::arr0(1.0).into_dyn());
        let z = goldstein(&x, &y);
        z.backward(false);

        assert_eq!(get_grad(&x), -5376.0);
        assert_eq!(get_grad(&y), 8064.0);
    }
}
