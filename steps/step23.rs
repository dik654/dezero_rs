// step23: 패키지로 정리
// 기존 step 파일에서 매번 중복되던 코드를 src/lib.rs 라이브러리로 추출
// 이제 use dezero::*; 로 가져다 쓴다
// Python의 dezero 패키지 구조 (dezero/__init__.py, dezero/core_simple.py)에 해당

use dezero::Variable;

#[cfg(test)]
mod tests {
    use super::*;

    fn get_val(v: &Variable) -> f64 {
        *v.data().first().unwrap()
    }

    fn get_grad(v: &Variable) -> f64 {
        *v.grad().unwrap().first().unwrap()
    }

    /// Python: y = (x + 3) ** 2, x = 1.0
    /// y = (1+3)^2 = 16, dy/dx = 2*(x+3) = 8
    #[test]
    fn test_package_example() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = (&x + 3.0).pow(2.0);
        y.backward(false, false);

        assert_eq!(get_val(&y), 16.0);
        assert_eq!(get_grad(&x), 8.0);
    }
}
