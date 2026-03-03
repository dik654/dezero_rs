// step67: GPT — 완전한 Decoder-only Transformer 언어 모델
//
// Step 61~66의 모든 부품을 최종 조립:
//   Token Embedding + Position Embedding
//   → TransformerBlock × N
//   → LayerNorm → Linear(vocab_size)
//   → logits (다음 토큰 확률 분포)
//
// 학습: softmax_cross_entropy로 다음 토큰 예측
// 생성: greedy argmax 자기회귀
//
// char-level 토이 예제로 학습/생성을 검증한다.

use dezero::{
    no_grad, reshape, softmax_cross_entropy_simple, sum, test_mode,
    AdamW, GPT, Variable,
};

#[cfg(test)]
mod tests {
    use super::*;

    // 간단한 char-level 토크나이저
    struct CharTokenizer {
        chars: Vec<char>,
    }

    impl CharTokenizer {
        fn from_text(text: &str) -> Self {
            let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
                .into_iter().collect();
            chars.sort();
            CharTokenizer { chars }
        }

        fn vocab_size(&self) -> usize { self.chars.len() }

        fn encode(&self, text: &str) -> Vec<usize> {
            text.chars().map(|c| {
                self.chars.iter().position(|&ch| ch == c).unwrap()
            }).collect()
        }

        fn decode(&self, tokens: &[usize]) -> String {
            tokens.iter().map(|&t| self.chars[t]).collect()
        }
    }

    fn make_idx(b: usize, t: usize, vocab_size: usize, seed: u64) -> Variable {
        let n = b * t;
        let mut rng = seed;
        let data: Vec<f64> = (0..n).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 % vocab_size as f64
        }).collect();
        Variable::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[b, t]), data).unwrap(),
        )
    }

    #[test]
    fn test_forward_shape() {
        // --- forward shape 검증 ---
        // vocab=10, D=8, H=2, N=2, block_size=16
        let gpt = GPT::new(10, 8, 2, 2, 16, 0.0, 42);
        let idx = make_idx(2, 4, 10, 0);
        let logits = gpt.forward(&idx);

        println!("input shape: {:?}", idx.shape());
        println!("logits shape: {:?}", logits.shape());
        assert_eq!(logits.shape(), vec![2, 4, 10]);
    }

    #[test]
    fn test_backward() {
        // --- backward: 모든 파라미터에 기울기 생성 ---
        let gpt = GPT::new(10, 8, 2, 2, 16, 0.0, 42);
        let idx = make_idx(1, 4, 10, 0);
        let logits = gpt.forward(&idx);

        // 다음 토큰 예측 loss
        let (b, t, v) = (1, 4, 10);
        let logits_2d = reshape(&logits, &[b * t, v]);
        let targets: Vec<usize> = vec![3, 5, 7, 1]; // 임의 타겟
        let loss = softmax_cross_entropy_simple(&logits_2d, &targets);
        loss.backward(false, false);

        let params = gpt.params();
        println!("param count: {}", params.len());
        for (i, p) in params.iter().enumerate() {
            assert!(p.grad().is_some(), "param {} has no grad", i);
        }
        // NaN/Inf 검사
        for (i, p) in params.iter().enumerate() {
            let g = p.grad().unwrap();
            assert!(g.iter().all(|v| v.is_finite()), "param {} grad has NaN/Inf", i);
        }
        println!("backward: all {} params have finite grads ✓", params.len());
    }

    #[test]
    fn test_causal_property() {
        // --- 인과성: 미래 토큰 변경 → 이전 logit 불변 ---
        let gpt = GPT::new(10, 8, 2, 2, 16, 0.0, 42);
        let _guard = no_grad();

        let idx1 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 4]),
                vec![1.0, 3.0, 5.0, 7.0],
            ).unwrap(),
        );
        let logits1 = gpt.forward(&idx1);
        let l1 = logits1.data();

        // 마지막 토큰만 변경
        let idx2 = Variable::new(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, 4]),
                vec![1.0, 3.0, 5.0, 2.0], // 7→2
            ).unwrap(),
        );
        let logits2 = gpt.forward(&idx2);
        let l2 = logits2.data();

        // t=0,1,2의 logits는 동일
        for t in 0..3 {
            for v in 0..10 {
                assert!(
                    (l1[[0, t, v]] - l2[[0, t, v]]).abs() < 1e-10,
                    "position {} vocab {} differs", t, v,
                );
            }
        }
        // t=3의 logits는 달라야 함
        let mut any_diff = false;
        for v in 0..10 {
            if (l1[[0, 3, v]] - l2[[0, 3, v]]).abs() > 1e-10 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "position 3 logits should differ");
        println!("Causal property verified ✓");
    }

    #[test]
    fn test_char_level_training() {
        // --- char-level 학습: "hello" 반복 학습 후 loss 감소 확인 ---
        let text = "hello world hello world hello world ";
        let tok = CharTokenizer::from_text(text);
        let tokens = tok.encode(text);
        let v = tok.vocab_size();

        println!("vocab: {:?}", tok.chars);
        println!("vocab_size: {}", v);
        println!("tokens: {:?}", &tokens[..10]);

        // 작은 GPT
        let gpt = GPT::new(v, 16, 2, 2, 32, 0.0, 42);
        let opt = AdamW::new(0.003, 0.0);

        // 시퀀스 길이 8로 학습
        let seq_len = 8;
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for epoch in 0..80 {
            let mut total_loss = 0.0;
            let mut count = 0;

            // 텍스트를 seq_len+1 크기 윈도우로 슬라이딩
            let mut offset = 0;
            while offset + seq_len + 1 <= tokens.len() {
                gpt.cleargrads();

                // 입력: tokens[offset..offset+seq_len]
                // 타겟: tokens[offset+1..offset+seq_len+1]
                let input: Vec<f64> = tokens[offset..offset + seq_len]
                    .iter().map(|&t| t as f64).collect();
                let target: Vec<usize> = tokens[offset + 1..offset + seq_len + 1].to_vec();

                let idx = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, seq_len]), input,
                    ).unwrap(),
                );
                let logits = gpt.forward(&idx); // (1, seq_len, V)
                let logits_2d = reshape(&logits, &[seq_len, v]);
                let loss = softmax_cross_entropy_simple(&logits_2d, &target);
                loss.backward(false, false);
                opt.update(&gpt.params());

                let loss_val: f64 = loss.data().iter().next().copied().unwrap();
                total_loss += loss_val;
                count += 1;
                offset += seq_len; // 비중복 슬라이딩
            }

            let avg_loss = total_loss / count as f64;
            if epoch == 0 { first_loss = avg_loss; }
            last_loss = avg_loss;

            if epoch < 3 || (epoch + 1) % 20 == 0 {
                println!("epoch {:3} | loss {:.4}", epoch + 1, avg_loss);
            }
        }

        println!("first loss: {:.4}, last loss: {:.4}", first_loss, last_loss);
        assert!(
            last_loss < first_loss * 0.5,
            "loss should decrease significantly: {} → {}",
            first_loss, last_loss,
        );
        println!("char-level training: loss decreased ✓");
    }

    #[test]
    fn test_text_generation() {
        // --- 학습 후 텍스트 생성 ---
        let text = "abcabcabcabcabcabc";
        let tok = CharTokenizer::from_text(text);
        let tokens = tok.encode(text);
        let v = tok.vocab_size();

        // 작은 GPT
        let gpt = GPT::new(v, 16, 2, 2, 32, 0.0, 42);
        let opt = AdamW::new(0.005, 0.0);

        // 패턴 "abc" 반복 학습
        let seq_len = 6;
        for _epoch in 0..150 {
            let mut offset = 0;
            while offset + seq_len + 1 <= tokens.len() {
                gpt.cleargrads();
                let input: Vec<f64> = tokens[offset..offset + seq_len]
                    .iter().map(|&t| t as f64).collect();
                let target: Vec<usize> = tokens[offset + 1..offset + seq_len + 1].to_vec();

                let idx = Variable::new(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[1, seq_len]), input,
                    ).unwrap(),
                );
                let logits = gpt.forward(&idx);
                let logits_2d = reshape(&logits, &[seq_len, v]);
                let loss = softmax_cross_entropy_simple(&logits_2d, &target);
                loss.backward(false, false);
                opt.update(&gpt.params());

                offset += seq_len;
            }
        }

        // 'a'로 시작하여 생성
        let start = tok.encode("a");
        let generated = gpt.generate(&start, 11);
        let generated_text = tok.decode(&generated);
        println!("generated: \"{}\"", generated_text);

        // "abc" 패턴이 포함되어 있는지 확인
        assert!(
            generated_text.contains("abc"),
            "generated text should contain 'abc' pattern: {}",
            generated_text,
        );
        println!("text generation verified ✓");
    }

    #[test]
    fn test_generate_respects_block_size() {
        // --- generate가 block_size를 초과하지 않는지 확인 ---
        // block_size=4인 매우 작은 모델
        let gpt = GPT::new(5, 8, 2, 1, 4, 0.0, 42);

        // 이미 block_size 이상의 토큰으로 시작해도 에러 없이 동작
        let start = vec![0, 1, 2, 3, 4, 0, 1]; // 7개 (> block_size=4)
        let result = gpt.generate(&start, 3);

        assert_eq!(result.len(), 10); // 7 + 3
        println!("generate with long context: {} tokens ✓", result.len());
    }

    #[test]
    fn test_cleargrads() {
        let gpt = GPT::new(10, 8, 2, 2, 16, 0.0, 42);
        let idx = make_idx(1, 4, 10, 0);

        let logits = gpt.forward(&idx);
        sum(&logits).backward(false, false);
        for p in gpt.params() {
            assert!(p.grad().is_some());
        }

        gpt.cleargrads();
        for p in gpt.params() {
            assert!(p.grad().is_none(), "grad should be cleared");
        }
        println!("cleargrads verified ✓");
    }
}
