// step57: im2col과 conv2d_simple
//
// step55에서 출력 크기 공식을 만들었고, step56은 이론 설명(코드 없음).
// step57에서 실제 합성곱 연산을 구현한다.
//
// 핵심 아이디어: im2col (image to column)
//   합성곱 = 슬라이딩 윈도우 × 내적 → 루프가 깊고 느림
//   im2col로 패치를 행렬로 펼치면 → 행렬 곱셈 한 번으로 대체 가능
//
//   (N, C, H, W)  →  im2col  →  (N*OH*OW, C*KH*KW)  ×  (C*KH*KW, OC)  =  (N*OH*OW, OC)
//    입력 이미지       패치 행렬        커널 가중치          합성곱 출력
//
// conv2d_simple: im2col + matmul 기반 합성곱
//   forward:  col = im2col(x) → y = col @ W^T → reshape to (N, OC, OH, OW)
//   backward: dcol = gy @ W → col2im → dx
//             dw = col^T @ gy → reshape to (OC, C, KH, KW)

use dezero::{conv2d_simple, im2col, Variable};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col() {
        // --- im2col 기본 ---
        // (1, 3, 7, 7) → kernel=5, stride=1, pad=0
        // OH = (7-5)/1+1 = 3, OW = 3
        // 출력: (1*3*3, 3*5*5) = (9, 75)
        let x1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 3, 7, 7]),
                (0..147).map(|i| i as f64 / 147.0).collect(),
            )
            .unwrap(),
        );
        let col1 = im2col(&x1, 5, 5, 1, 1, 0, 0);
        println!("col1.shape = {:?}", col1.shape());
        assert_eq!(col1.shape(), vec![9, 75]);

        // --- im2col 배치 ---
        // (10, 3, 7, 7) → 같은 커널
        // 출력: (10*3*3, 3*5*5) = (90, 75)
        let x2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[10, 3, 7, 7]),
                (0..1470).map(|i| i as f64 / 1470.0).collect(),
            )
            .unwrap(),
        );
        let col2 = im2col(&x2, 5, 5, 1, 1, 0, 0);
        println!("col2.shape = {:?}", col2.shape());
        assert_eq!(col2.shape(), vec![90, 75]);
    }

    #[test]
    fn test_conv2d_simple() {
        // --- conv2d forward + backward ---
        // x: (1, 5, 15, 15), W: (8, 5, 3, 3), stride=1, pad=1
        // OH = (15+2-3)/1+1 = 15, OW = 15
        // y: (1, 8, 15, 15)
        let n = 1;
        let c = 5;
        let h = 15;
        let w_size = 15;
        let oc = 8;
        let kh = 3;
        let kw = 3;

        let x = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[n, c, h, w_size]),
                (0..n * c * h * w_size).map(|i| (i as f64 - 500.0) / 500.0).collect(),
            )
            .unwrap(),
        );
        let w = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[oc, c, kh, kw]),
                (0..oc * c * kh * kw).map(|i| (i as f64 - 180.0) / 180.0).collect(),
            )
            .unwrap(),
        );

        let y = conv2d_simple(&x, &w, None, 1, 1);
        println!("y.shape = {:?}", y.shape());
        assert_eq!(y.shape(), vec![1, 8, 15, 15]);

        // backward
        y.backward(false, false);
        let x_grad = x.grad().unwrap();
        println!("x.grad.shape = {:?}", x_grad.shape());
        assert_eq!(x_grad.shape(), &[1, 5, 15, 15]);

        println!("All conv2d tests passed!");
    }
}
