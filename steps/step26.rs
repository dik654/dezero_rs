// step26: 계산 그래프 시각화
// Graphviz의 DOT 언어로 계산 그래프를 출력한다
// Python의 dezero.utils.plot_dot_graph에 해당

use dezero::{get_dot_graph, plot_dot_graph, Variable};

fn goldstein(x: &Variable, y: &Variable) -> Variable {
    let sum_xy = x + y;
    let a = &sum_xy + 1.0;

    let b1 = 19.0 - &(14.0 * x);
    let b2 = &b1 + &(3.0 * &x.pow(2.0));
    let b3 = &b2 - &(14.0 * y);
    let b4 = &b3 + &(6.0 * &(x * y));
    let b = &b4 + &(3.0 * &y.pow(2.0));

    let a2b = &a.pow(2.0) * &b;
    let part1 = 1.0 + &a2b;

    let c = &(2.0 * x) - &(3.0 * y);

    let d1 = 18.0 - &(32.0 * x);
    let d2 = &d1 + &(12.0 * &x.pow(2.0));
    let d3 = &d2 + &(48.0 * y);
    let d4 = &d3 - &(36.0 * &(x * y));
    let d = &d4 + &(27.0 * &y.pow(2.0));

    let c2d = &c.pow(2.0) * &d;
    let part2 = 30.0 + &c2d;

    &part1 * &part2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DOT 그래프 문자열이 올바른 구조를 가지는지 검증
    #[test]
    fn test_dot_graph() {
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        let y = Variable::new(ndarray::arr0(1.0).into_dyn());
        let z = goldstein(&x, &y);
        z.backward(false, false);

        x.set_name("x");
        y.set_name("y");
        z.set_name("z");

        let dot = get_dot_graph(&z, false);

        // DOT 형식 기본 구조 확인
        assert!(dot.starts_with("digraph g {"));
        assert!(dot.ends_with("}"));

        // 변수 노드 존재
        assert!(dot.contains("label=\"x\""));
        assert!(dot.contains("label=\"y\""));
        assert!(dot.contains("label=\"z\""));

        // 함수 노드 존재 (Goldstein-Price는 Add, Sub, Mul, Pow 사용)
        assert!(dot.contains("label=\"Add\""));
        assert!(dot.contains("label=\"Mul\""));
        assert!(dot.contains("label=\"Pow\""));
        assert!(dot.contains("label=\"Sub\""));

        // 엣지(->)가 존재
        assert!(dot.contains("->"));
    }

    /// verbose 모드에서 형상 정보가 포함되는지 검증
    #[test]
    fn test_dot_graph_verbose() {
        let x = Variable::with_name(ndarray::arr0(1.0).into_dyn(), "x");
        let y = &x + 1.0;
        let dot = get_dot_graph(&y, true);

        // verbose 모드: 이름 + 형상 + dtype
        assert!(dot.contains("x: [] f64"));
    }

    /// Graphviz가 설치되어 있으면 PNG 파일 생성 검증
    #[test]
    fn test_plot_dot_graph() {
        let x = Variable::with_name(ndarray::arr0(1.0).into_dyn(), "x");
        let y = Variable::with_name(ndarray::arr0(1.0).into_dyn(), "y");
        let z = goldstein(&x, &y);
        z.set_name("z");

        let result = plot_dot_graph(&z, false, "goldstein.png");
        if result.is_ok() {
            // 파일이 생성되었는지 확인 후 정리
            assert!(std::path::Path::new("goldstein.png").exists());
            let _ = std::fs::remove_file("goldstein.png");
            let _ = std::fs::remove_file("goldstein.dot");
        }
        // Graphviz 미설치 시에도 테스트 실패하지 않음
    }
}
