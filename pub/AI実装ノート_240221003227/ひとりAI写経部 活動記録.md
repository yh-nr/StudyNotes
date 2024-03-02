---
marp: true
headingDivider: 2
# theme: portrait_A4
# theme: portrait_4to3
theme: portrait_16to9
# theme: landscape_A4
# theme: landscape_4to3
# theme: landscape_16to9
# theme: _styletest
paginate: true
math: katex # Latexを書けるよう設定
---


<!-- _class: lead -->
<!-- _paginate: false -->
# ひとりＡＩ写経部

## 目的

E資格の勉強で実装があまりできていないので、ゼロつくや論文の実装を通して実装力を身に付ける。

### AI実装

- ゼロつく実装[①](
https://github.com/oreilly-japan/deep-learning-from-scratch)[②](https://github.com/oreilly-japan/deep-learning-from-scratch-2)[③](https://github.com/oreilly-japan/deep-learning-from-scratch-3)[④](https://github.com/oreilly-japan/deep-learning-from-scratch-4)[⑤](https://github.com/oreilly-japan/deep-learning-from-scratch-5)
- 強化学習実装
- [AI実装検定](https://kentei.ai/)
  - 合格体験記的なやつを調査
  - 論文リストアップ
    - NLP
      - [seq2seq](https://arxiv.org/abs/1409.3215)
      - Transformer
      - HRED
      - Word2Vec (Skip-gram)
    - Model
      - VGG
      - GoogLeNet
      - ResNet/WideResNet
      - MobileNet
      - EfficientNet
      - DenseNet

## ゼロつく2章

#### パーセプトロン

- パーセプトロン
- 人口ニューロン
- 単純パーセプトロン

$$
y =
\begin{cases}
0 & (w_1x_1 + w_2x_2 \le \theta) \\
1 & (w_1x_1 + w_2x_2 > \theta) \\
\end{cases}
$$
