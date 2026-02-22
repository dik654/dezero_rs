// step47: softmax와 cross-entropy (분류 문제)
//
// step46까지는 회귀(regression): 연속적인 값을 예측 (y = sin(2πx))
// step47부터는 분류(classification): 입력이 어떤 클래스에 속하는지 예측
//
// 회귀 vs 분류:
//   회귀: 출력 = 실수값 (예: 3.14), 손실 = MSE
//   분류: 출력 = 각 클래스의 확률 (예: [0.1, 0.7, 0.2]), 손실 = cross-entropy
//
// softmax가 필요한 이유:
//   모델의 원시 출력(logit)은 [-∞, +∞] 범위의 임의의 실수
//   예: [2.1, -0.3, 0.8] → 이대로는 "확률"이 아님
//   softmax 변환: exp(x_i) / sum(exp(x_j))
//   → [0.72, 0.07, 0.21] → 합이 1, 각 값이 0~1 → 확률로 해석 가능
//
// cross-entropy가 필요한 이유:
//   정답이 클래스 2이면, p[2]가 1에 가까울수록 좋다
//   -log(p[2]): p[2]=1이면 0 (완벽), p[2]=0.01이면 4.6 (매우 나쁨)
//   이것이 cross-entropy 손실

use dezero::{
    exp, softmax_cross_entropy_simple, softmax_simple, sum, Model, Variable, MLP,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// 1차원 softmax (step47 Python 코드의 softmax1d)
    /// 전체 합으로 나누는 단순 버전
    fn softmax1d(x: &Variable) -> Variable {
        let y = exp(x);
        let sum_y = sum(&y);
        &y / &sum_y
    }

    #[test]
    fn test_softmax_and_cross_entropy() {
        // 모델: 입력 2 → 은닉 10 → 출력 3(클래스 수)
        let model = MLP::new(&[10, 3]);

        // --- 단일 샘플로 softmax1d 테스트 ---
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![0.2, -0.4]).unwrap(),
        );
        let y = model.forward(&x);
        let p = softmax1d(&y);
        println!("y = {}", y);
        println!("softmax1d(y) = {}", p);

        // softmax 결과의 합은 1이어야 함
        let p_sum: f64 = p.data().iter().sum();
        assert!(
            (p_sum - 1.0).abs() < 1e-6,
            "softmax sum should be 1.0, got {}",
            p_sum
        );

        // --- 배치 softmax 테스트 ---
        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[4, 2]),
                vec![0.2, -0.4, 0.3, 0.5, 1.3, -3.2, 2.1, 0.3],
            )
            .unwrap(),
        );
        // 정답 클래스: 샘플0→클래스2, 샘플1→클래스0, 샘플2→클래스1, 샘플3→클래스0
        let t: Vec<usize> = vec![2, 0, 1, 0];

        let y = model.forward(&x);
        let p = softmax_simple(&y);
        println!("\ny (logits) = {}", y);
        println!("softmax(y) = {}", p);

        // 각 행의 합이 1인지 확인
        let p_data = p.data();
        for i in 0..4 {
            let row_sum: f64 = (0..3).map(|j| p_data[[i, j]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {} softmax sum should be 1.0, got {}",
                i,
                row_sum
            );
        }

        // --- cross-entropy 손실 및 backward ---
        let loss = softmax_cross_entropy_simple(&y, &t);
        println!("\ncross-entropy loss = {}", loss);

        let loss_val = loss.data().iter().next().copied().unwrap();
        assert!(loss_val > 0.0, "cross-entropy loss should be positive");

        // backward
        model.cleargrads();
        loss.backward(false, false);

        // 모든 파라미터에 gradient가 있는지 확인
        for p in model.params() {
            assert!(
                p.grad().is_some(),
                "all parameters should have gradients after backward"
            );
        }
        println!("backward completed successfully");
    }
}
