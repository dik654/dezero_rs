// step55: CNN — 합성곱 출력 크기 계산
//
// CNN(Convolutional Neural Network)의 기초: 합성곱 레이어의 출력 크기를 결정하는 공식.
//
// 합성곱 연산의 출력 크기:
//   OH = (H + 2*PH - KH) / SH + 1
//   OW = (W + 2*PW - KW) / SW + 1
//
// 각 파라미터:
//   H, W:    입력 크기 (높이, 너비)
//   KH, KW:  커널(필터) 크기
//   SH, SW:  스트라이드 (커널이 이동하는 칸 수)
//   PH, PW:  패딩 (입력 테두리에 추가하는 0의 개수)
//
// 예) 4×4 입력, 3×3 커널, stride=1, pad=1:
//   OH = (4 + 2*1 - 3) / 1 + 1 = 4
//   → 패딩 1을 주면 출력 크기가 입력과 동일 ("same" padding)

use dezero::get_conv_outsize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_outsize() {
        let (h, w) = (4, 4);       // 입력 크기
        let (kh, kw) = (3, 3);     // 커널 크기
        let (sh, sw) = (1, 1);     // 스트라이드
        let (ph, pw) = (1, 1);     // 패딩

        let oh = get_conv_outsize(h, kh, sh, ph);
        let ow = get_conv_outsize(w, kw, sw, pw);
        println!("input: {}x{}, kernel: {}x{}, stride: {}x{}, pad: {}x{}", h, w, kh, kw, sh, sw, ph, pw);
        println!("output: {}x{}", oh, ow);

        // pad=1, stride=1, kernel=3 → 출력 = 입력 (same padding)
        assert_eq!(oh, 4);
        assert_eq!(ow, 4);

        // --- 다양한 케이스 ---

        // pad=0, stride=1, kernel=3 → 출력이 2 줄어듦 (valid padding)
        let oh = get_conv_outsize(4, 3, 1, 0);
        assert_eq!(oh, 2); // (4 + 0 - 3) / 1 + 1 = 2

        // stride=2 → 출력이 절반
        let oh = get_conv_outsize(8, 3, 2, 1);
        assert_eq!(oh, 4); // (8 + 2 - 3) / 2 + 1 = 4

        // 7×7 입력, 5×5 커널, stride=1, pad=0 → im2col의 출력 행 수
        let oh = get_conv_outsize(7, 5, 1, 0);
        assert_eq!(oh, 3); // (7 + 0 - 5) / 1 + 1 = 3

        println!("All conv outsize tests passed!");
    }
}
