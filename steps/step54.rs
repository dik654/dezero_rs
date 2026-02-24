// step54: Dropout 정규화
//
// step53까지의 문제:
//   MLP가 학습 데이터를 "암기"할 수 있음 → 과적합(overfitting)
//   특히 파라미터가 많은 모델(795,010개)에서 심각
//
// Dropout (Srivastava et al., 2014):
//   훈련 시 각 뉴런을 확률 p로 무작위 비활성화
//   → 매번 다른 서브 네트워크로 학습하는 효과
//   → 특정 뉴런에 의존하지 않는 강건한 특징 학습
//   → 앙상블(ensemble) 학습의 근사로 해석 가능
//
// Inverted Dropout:
//   훈련: y = x * mask / (1 - p)
//     mask: 각 원소가 확률 p로 0, (1-p)로 1
//     1/(1-p) 스케일링: 추론 시 보정 불필요하게 만듦
//   추론: y = x  (그대로 통과)
//
// Python: with test_mode(): → dezero.Config.train = False
// Rust:   let _guard = test_mode(); → TRAINING thread_local = false
//         RAII 가드: 스코프 종료 시 자동 복원

use dezero::{dropout, test_mode, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout() {
        let x = Variable::new(ndarray::arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn());
        println!("x = {:?}", x.data());

        // --- 훈련 모드 (기본) ---
        // 일부 원소가 0, 나머지는 1/(1-0.5) = 2.0으로 스케일링
        let y = dropout(&x, 0.5);
        println!("train mode: y = {:?}", y.data());

        // 스케일링 확인: 0이 아닌 원소는 2.0이어야 함 (inverted dropout)
        for &val in y.data().iter() {
            assert!(
                val == 0.0 || (val - 2.0).abs() < 1e-10,
                "expected 0.0 or 2.0, got {}",
                val
            );
        }

        // --- 추론 모드 ---
        // test_mode() 가드: TRAINING = false → dropout 비활성화
        {
            let _guard = test_mode();
            let y = dropout(&x, 0.5);
            println!("test mode:  y = {:?}", y.data());

            // 추론 시에는 입력이 그대로 통과
            for &val in y.data().iter() {
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "test mode should pass through, got {}",
                    val
                );
            }
        }
        // _guard가 Drop되면서 TRAINING = true로 자동 복원

        // --- 훈련 모드 다시 확인 ---
        let y = dropout(&x, 0.5);
        println!("train mode (restored): y = {:?}", y.data());
        // 다시 dropout이 적용되어야 함
        let has_zero = y.data().iter().any(|&v| v == 0.0) || y.data().iter().all(|&v| (v - 2.0).abs() < 1e-10);
        assert!(has_zero, "dropout should be active after test_mode guard is dropped");

        println!("All dropout tests passed!");
    }
}
