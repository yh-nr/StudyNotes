<!-- 
marp: true
headingDivider: 1
theme: A4-Manual
paginate: true
footer: E資格範囲の数式まとめ
-->


---


$$\[ A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}, \quad B = \begin{pmatrix} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \\ b_{31} & b_{32} & b_{33} \end{pmatrix} \]$$


$$
\[ AB = \begin{pmatrix} a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} & a_{11}b_{12} + a_{12}b_{22} + a_{13}b_{32} & a_{11}b_{13} + a_{12}b_{23} + a_{13}b_{33} \\ a_{21}b_{11} + a_{22}b_{21} + a_{23}b_{31} & a_{21}b_{12} + a_{22}b_{22} + a_{23}b_{32} & a_{21}b_{13} + a_{22}b_{23} + a_{23}b_{33} \\ a_{31}b_{11} + a_{32}b_{21} + a_{33}b_{31} & a_{31}b_{12} + a_{32}b_{22} + a_{33}b_{32} & a_{31}b_{13} + a_{32}b_{23} + a_{33}b_{33} \end{pmatrix} \]
$$

# 固有方程式  
<!-- _header: 線形代数 -->

$$ \large \det (\lambda I - A) = 0$$
###### 各要素の意味など
- $\det$　は行列式(determinant) を意味する。 <br> $\left| \begin{array}{cc} \lambda I - A \end{array} \right| =0$　と表現される事もある。
- $\lambda$　は固有値
- $I$　は単位行列
- $A$　はn次正方行列




# 特異値分解

$$ \large A = U\Sigma V^T $$
###### 各要素の意味など
- $A$　は
- $U$　は
- $\Sigma$　は
- $V^T$　は
- 




# ベルヌーイ分布

$$ \large f(x;p) = p^x(1-p)^{1-x} $$
###### 各要素の意味など
- $x$　は成功か失敗を表す変数（$k$で表されることもある？）
- $p$　は単一試行での成功確率
- $f(x;p)$　はベルヌーイ分布の**確率質量関数**で、パラメータ$p$が与えられた時の変数$x$の関数という意味
	- $x=1$ の場合、 $p^x=p$ となり $(1-p)^{1-x} = 1$ となるため、<center>$f(x;p)=p$</center>
	- $x=0$ の場合、 $p^x=1$ となり $(1-p)^{1-x} = 1-p$ となるため、<center>$f(x;p)=1-p$</center>
- $p$の$x$乗の様な表現はただのモデル化であり物理的プロセスや現象を直接表現しているわけではない？（「こうするとうまく表現できる」以上の深い意味はない？）




# ベルヌーイ分布の期待値

$$\large\begin{align}\mathbb{E}[X] &= \sum_{x=0}^{1}xp^x(1-p)^{1-x} \\\\&= p\end{align}$$
###### 各要素の意味など
- $\mathbb{E}[X]$は



# ベルヌーイ分布の分散

$$\large\begin{align}Var[X] &= \mathbb{E}[(X-\mathbb{E}[X])^2)] \\
  Var[X] &= \mathbb{E}[X^2]-\mathbb{E}[X]^2 \\
  &= \sum_{x=0}^{1}x^2p^x(1-p)^{1-x}-p^2 \\
  &= p-p^2\\&=p(1-p)\end{align}$$
###### 各要素の意味など
- 



# マルチヌーイ分布（カテゴリ分布）

$$\large{f(x;p) = \prod_{j=1}^{k}p_j^{x_j}}　\\(ただし、\sum_{j=1}^{k}p_j =1、0\le p_j \le 1、j=1,....)$$

###### 各要素の意味など
- 



# マルチヌーイ分布の負の対数尤度<br>（カテゴリ分布）

$$\large\begin{align} -\log L_D (p) &= -\log\prod_{i=1}^{n}f(x_i;p)　 \\
  &= -\sum_{i=1}^{n}\log\prod_{j=1}^{k}p_j^{x_{ij}} \\
  &= -\sum_{i=1}^{n}\sum_{j=1}^{k}\log p_j^{x_{ij}} \\
  &= -\sum_{i=1}^{n}\sum_{j=1}^{k}x_{ij}\log p_j
  \end{align}$$
###### 各要素の意味など
- 



# 正規分布

$$\large f(x; \mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(x-\mu)^2\right)$$
###### 各要素の意味など
- 



# 正規分布の負の対数尤度

$$\large\begin{align} L(\mu) &= \prod_{i=1}^{n}f(x_i;\mu)\\
&=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(x_i-\mu)^2\right)\\
 -\log L(\mu) &= -\log \left(\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(x_i-\mu)^2\right)\right)\\
&=-\sum_{i=1}^{n}\log\left(\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(x_i-\mu)^2\right)\right)\\
&=-\sum_{i=1}^{n}\left(\log\left(\frac{1}{\sqrt{2\pi}}\right)-\frac{1}{2}(x_i-\mu)^2\right)\\
&=-n\log\left(\frac{1}{\sqrt{2\pi}}\right)-\frac{1}{2}\sum_{i=1}^{n}(x_i-\mu)^2\\
\end{align}$$
###### 各要素の意味など
- 



# 正規分布の最尤推定
$$\large\begin{align}
\frac{d}{d\mu}g(\mu) &= \frac{1}{2}\sum_{i=1}^{n}\frac{d}{d\mu}(x_i-\mu)^2\\
&= \frac{1}{2}\sum_{i=1}^{n}(-2(x_i-\mu))\\
&= \sum_{i=1}^{n}\mu-\sum_{i=1}^{n}x_i\\
&= n\mu-\sum_{i=1}^{n}x_i\\
\end{align}$$

$$\large\hat\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$$


###### 各要素の意味など
- 



# エントロピー
$$\large H(X) = -\sum_{x}p(x)\log_2p(x)$$

###### 各要素の意味など
- 



# 交差エントロピー(クロスエントロピー)の定義
$$\large H(p,q) = -\sum_{x}p(x)\log_2q(x)$$
###### 各要素の意味など
- $p(x)$　真の(正解の)確率分布
- $q(x)$　推定したモデルの確率分布






# 二値交差エントロピー(バイナリクロスエントロピー)　※1/8追加※
$$\large D_{BC} = -P(x=0)\log Q(x=0)-(1-P(x=0))\log(1-Q(x-0))$$
###### 各要素の意味など
- $p(x)$　真の(正解の)確率分布
- $q(x)$　推定したモデルの確率分布





# KLダイバージェンス
$$\large D(p||q) = \sum_{x}p(x)\log_2\frac{p(x)}{q(x)}$$



###### 各要素の意味など
- 



# JSダイバージェンス
$$\large D_{JS}(p||q) = \frac{1}{2}\left(\sum_{x}p(x)\log_2\frac{p(x)}{r(x)}+\sum_{x}q(x)\log_2\frac{q(x)}{r(x)}\right)$$
$$\large r(x) = \frac{p(x)+q(x)}{2}$$



###### 各要素の意味など
- 



# ベイズの定理
$$\large p(C|x) = \frac{p(x|C)p(C)}{p(x)}$$



###### 各要素の意味など
- 



# バイアス・バリアンス・ノイズ
$$\large\mathbb{E}(L) = \int\{y(x)-h(x)\}^2p(x)dx +\iint\{h(x)-t\}^2p(x,t)dxdt$$

$$\large\begin{align}
&\int\{\mathbb{E}_D[y(x;D)] - h(x)\}^2p(x)dx \\
&\int\mathbb{E}_D[\{y(x;D) - \mathbb{E}_D[y(x;D)]\}^2p(x)dx \\
&\iint\{h(x) - t\}^2p(x,t)dxdt \\
\end{align}$$



###### 各要素の意味など
- 



# シグモイド関数
$$\large f(x) = \frac{1}{1+\exp(-x)}$$



###### 各要素の意味など
- $\exp(-x)$ は $e^{-x}$ の意、x=0の時1となるため、$f(x)=\frac{1}{2}$となる
- 



# ReLU
$$\large f(x) = \max(0,x)$$



###### 各要素の意味など
- 



# オッズ
$$\large \frac{p(y=1|x)}{p(y=0|x)}=\frac{\hat y}{1-\hat y}$$

$$\large\begin{align}
\frac{\hat y}{1-\hat y} &= \frac{\frac{1}{1+\exp(-w^Tx-b)}}{1-\frac{1}{1+\exp(-w^Tx-b)}}\\
&= \frac{1}{(1+\exp(-w^Tx-b))-1}\\
&= \frac{1}{\exp(-w^Tx-b)}\\
&= \exp(w^Tx-b)
\end{align}$$



###### 各要素の意味など
- 



# ガウスカーネル
$$\large k(x,x')=\exp\left(-\frac{||x-x'||^2}{\beta}\right)$$



###### 各要素の意味など
- 



# 正則化
$$\large\begin{align}
&E+\lambda_2||w||_2^2 \\
&E+\lambda_1||w||_1 \\
&E+\lambda_1||w||_1+ \lambda_2||w||_2^2
\end{align}$$



###### 各要素の意味など
- 



# ソフトマックス
$$\large \text{softmax}(z)_i=\frac{\exp(z_i)}{\Sigma_j\exp(z_j)}$$



Numpy表記：``` np.exp(z) / np.sum(np.exp(z))```

###### 各要素の意味など
- 



# 二乗和誤差
$$\large \frac{1}{2}\sum_{k=1}^{K}=(y_k-t_k)^2$$



###### 各要素の意味など
- 



# 生成モデル
$$\large p(y|x)p(x) = \frac{p(x,y)}{p(x)}\cdot p(x) = p(x,y)$$



###### 各要素の意味など
- 



# ベルマン方程式
$$\large\begin{align}
V^\pi(s) &= \mathbb{E}[G_t|S_t = s]\\\\
&= \mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_t = s]\\\\
&= \sum_a\pi(a|s)\sum_{s',r}P(s',r|s,a)[r+\gamma\mathbb{E}_\pi[G_{t+1}|S_{t+1} = s']]\\\\
&= \sum_a\pi(a|s)\sum_{s',r}P(s',r|s,a)[r+\gamma V^\pi(s')]
\end{align}$$

$$\large\begin{align}
Q^\pi(s,a) &= \mathbb{E}[R_{t+1}+\gamma V^\pi(S_{t+1})|S_t = s,A_t = a]\\
&= \sum_{s',r}P(s',r|s,a)[r+\gamma V^\pi(s')]
\end{align}$$

###### 各要素の意味など
- 



# SARSA
$$\large Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)] $$


###### 各要素の意味など
- 



# Q学習
$$\large Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1} + \gamma \max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\right] $$


###### 各要素の意味など
- 



# 方策勾配定理
$$\large\begin{align}
&\nabla_\theta J(\theta) = \sum d^{\pi_\theta}(s) \sum \nabla_\theta\pi_\theta(a|s,\theta)Q^{\pi_\theta}(s,a) \\
&\nabla_\theta\log \pi_\theta(a|s) = \frac{\partial\pi_\theta(a|s)}{\partial\theta}\frac{1}{\pi_\theta(a|s)} \\
&d^{\pi_\theta}(s) = \sum_{k=0}^{\infty}\gamma^kP^{\pi_\theta}(s_k=s|s_0) 
\end{align}$$
$$\large\begin{align}
\nabla_\theta J(\theta) &= \sum d^{\pi_\theta}(s) \sum_a( \nabla_\theta\pi_\theta(a|s,\theta))Q^{\pi_\theta}(s,a) \\
&= \sum d^{\pi_\theta}(s) \sum_a\pi_\theta(a|s,\theta)( \nabla_\theta\log\pi_\theta(a|s,\theta))Q^{\pi_\theta}(s,a) \\
&= \mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta(a|s,\theta)Q^{\pi_\theta}(s,a)] \\
\end{align}$$


###### 各要素の意味など
- 



# モンテカルロ近似
$$\large
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[f(s,a)] \approx \frac{1}{N}\sum_{n=1}^{N}\frac{1}{T}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t^n|s_t^n)Q^{\pi_\theta}(s_t^n, a_t^n)
$$


###### 各要素の意味など
- 



# 交差エントロピー誤差
$$\large E = -\frac{1}{N}\sum_{n}\sum_{k}t_{nk}\log y_{nk}$$


###### 各要素の意味など
- $N$：データ個数
- $k$：データの次元数



# アフィンレイヤ
$$\large H = XW + B$$
$$\large \frac{\partial L}{\partial W} = X^T\frac{\partial L}{\partial H}$$


###### 各要素の意味など
- 



# モーメンタム
$$\large v_{t+1} = av_t - \eta\frac{\partial L}{\partial \theta_t}$$
$$\large \theta_{t+1} = \theta_t+v_{t+1}$$


###### 各要素の意味など
- 



# 確率的勾配降下法　※1/8追加
$$$$


###### 各要素の意味など
- 



# NesterovAG
$$\large v_{t+1} = av_t - \eta\frac{\partial L}{\partial (\theta_t + av_t)}$$
$$\large \theta_{t+1} = \theta_t+v_{t+1}$$


###### 各要素の意味など
- 



# AdaGrad
$$\large h_{t+1} = h_t + \frac{\partial L}{\partial \theta_t}\odot \frac{\partial L}{\partial \theta_t}$$
$$\large \theta_{t+1} = \theta_t - \eta\frac{1}{\varepsilon+\sqrt{h_{t+1}}}\odot \frac{\partial L}{\partial \theta_t}$$


###### 各要素の意味など
- 



# RMSProp
$$\large h_{t+1} = \rho h_t + (1-\rho)\frac{\partial L}{\partial \theta_t}\odot \frac{\partial L}{\partial \theta_t}$$
$$\large \theta_{t+1} = \theta_t - \eta\frac{1}{\sqrt{\varepsilon+h_{t+1}}}\odot \frac{\partial L}{\partial \theta_t}$$


###### 各要素の意味など
- 



# Adam
$$\large\begin{align}
&m_{t+1} = \rho_1 m_t + (1-\rho_1)\frac{\partial L}{\partial \theta_t}\\
&v_{t+1} = \rho_2 v_t + (1-\rho_2)\frac{\partial L}{\partial \theta_t}\odot \frac{\partial L}{\partial \theta_t}\\\\
&\hat m_{t+1} = \frac{m_{t+1}}{1-\rho_1^t}\\
&\hat v_{t+1} = \frac{v_{t+1}}{1-\rho_2^t}\\\\
&\theta_{t+1} = \theta_t - \eta\frac{1}{\sqrt{\hat v_{t+1}}+\varepsilon}\odot \hat m_{t+1}
\end{align}$$

###### 各要素の意味など
- 



# バッチ正規化
$$\large\begin{align}
&h' = \frac{h-\mu}{\sigma}\\
&\gamma h' + \beta
\end{align}$$

###### 各要素の意味など
- 



# 畳み込み
$$\large(I*K)(i,j)=\sum_m\sum_nI(i+m,j+n)K(m,n)$$

###### 各要素の意味など
- 



# IoU
$$ IoU(B_{true},B_{pred})=\frac{|B_{true}\cap B_{pred}|}{|B_{true}\cup B_{pred}|}=\frac{|B_{true}\cap B_{pred}|}{|B_{true}|+|B_{pred}|-|B_{true}\cap B_{pred}|}$$

###### 各要素の意味など
- 



# Dice 係数
$$\large Dice(S_{true},S_{pred})=\frac{|S_{true}\cap S_{pred}|}{\frac{|S_{true}|+|S_{pred}|}{2}}=\frac{2|S_{true}\cap S_{pred}|}{|S_{true}|+|S_{pred}|}$$

###### 各要素の意味など
- 



# AP
$$\large AP = \frac{1}{11}\sum_{r\in\{0,0.1,0.2,...,1\}}p_{interp}(r)$$
$$\large p_{interp}(r)=\max_{\tilde r\ge r}p(\tilde r)$$

###### 各要素の意味など
- 



# LSTMの順伝播
$$\large\begin{align}
&G = \text{tanh}\left(X_tW_x^{(g)}+H_{t-1}W_h^{(g)}+b^{(g)}\right)\\
&I = \text{sigmoid}\left(X_tW_x^{(i)}+H_{t-1}W_h^{(i)}+b^{(i)}\right)\\
&F = \text{sigmoid}\left(X_tW_x^{(f)}+H_{t-1}W_h^{(f)}+b^{(f)}\right)\\
&O = \text{sigmoid}\left(X_tW_x^{(o)}+H_{t-1}W_h^{(o)}+b^{(o)}\right)
\end{align}$$

###### 各要素の意味など
- 



# GRUの順伝播
$$\large\begin{align}
&R = \text{sigmoid}\left(X_tW_x^{(r)}+H_{t-1}W_h^{(r)}+b^{(r)}\right)\\
&Z = \text{sigmoid}\left(X_tW_x^{(z)}+H_{t-1}W_h^{(z)}+b^{(z)}\right)\\\\
&\tilde H = \text{tanh}\left\{X_tW_x^{(\tilde h)}+(R\odot H_{t-1})W_h^{(\tilde h)}+b^{(\tilde h)}\right\}\\\\
&H_t = Z\odot H_{t-1}+(1-Z)\odot\tilde H
\end{align}$$

###### 各要素の意味など
- 



# WaveNetの定式化
$$\large
p(x) = \prod_{t=1}^{T}p(x_t|x_1,x_2,...,x_{t-1})\\
$$

###### 各要素の意味など
- 



# TransformerのScaled Dot-Product Attention
$$\large\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

###### 各要素の意味など
- 



# TransformerのPositional Encoding
$$\large\begin{align}
&PE_(pos,2i) = \text{sin}\left(pos/10000^{2i/d_{model}}\right)\\
&PE_(pos,2i+1) = \text{cos}\left(pos/10000^{2i/d_{model}}\right)
\end{align}$$

###### 各要素の意味など
- 



# VAEの損失関数
$$\begin{align}
&-\log p(x) \le -L = \mathbb{E}_{z\sim p(z|x)}[-\log p(x|z)] + \int\log\left(\frac{p(x|z)}{p(z)}\right)p(z|x)dz
\end{align}$$

###### 各要素の意味など
- 



# GANの定式化
$$\large
\min_G\max_D\mathbb{E_x}[\log D(x)]+\mathbb{E_z}[\log(1-D(G(z)))]
$$

###### 各要素の意味など
- 



# DQN
$$\large
L(\theta)=\mathbb{E_{s,a,r,s'\sim D}}\left[\left(r+\gamma\max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta)\right)^2\right]
$$

###### 各要素の意味など
- 



# 蒸留における温度付きソフトマックス関数
$$\large\text{Softmax}(z)_i=\frac{\exp(z_i/T)}{\Sigma_j\exp(z_j/T)}$$

###### 各要素の意味など
- 

