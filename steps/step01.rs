use ndarray::ArrayD;

struct Variable {
    data: ArrayD<f64>,
}

// 초기화 함수
impl Variable {
    // 인수를 인스턴스 변수에 저장
    // 머신 러닝 시스템은 기본 데이터 구조로 다차원 배열 사용
    fn new(data: ArrayD<f64>) -> Self {
        Variable { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_variable() {
        // 0차원 스칼라값 1을 인스턴스에 넣기
        let x = Variable::new(ndarray::arr0(1.0).into_dyn());
        // 데이터가 제대로 담겨있는지 확인
        assert_eq!(x.data, ndarray::arr0(1.0).into_dyn());
    }

    #[test]
    fn test_insert_new_data() {
        // 0차원 스칼라값 1을 인스턴스에 넣기
        // 인스턴스의 변수가 바뀔 것이므로 가변(mut) 선언
        let mut x = Variable::new(ndarray::arr0(1.0).into_dyn());
        // 인스턴스에 새로운 데이터 대입
        x.data = ndarray::arr0(2.0).into_dyn();
        // 새로운 데이터가 대입되었는지 확인
        assert_eq!(x.data, ndarray::arr0(2.0).into_dyn());
    }
}
