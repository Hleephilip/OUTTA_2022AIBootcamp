# [ML/DL] Quiz Solution : Neural Networks

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)

## Problem 1.
은닉층의 개수가 2개 이상인 신경망을 무엇이라고 부르는가?


## Solution 1.

[정답] 심층 신경망


<br>

## Problem 2.

아래 내용 중 **올바른 문장**의 개수는 몇 개인가?

(1) 벡터화 과정의 문제점을 해결하기 위해 합성곱 신경망을 사용할 수 있다.

(2) Sigmoid 함수에서 Gradient Vanishing 이란 역전파가 진행될수록 기울기 값이 로 발산하게 되는 문제로, tanh 함수를 사용해 해결할 수 있다.

(3) 기울기 폭주 현상의 경우 Gradient Clipping을 통해 해결할 수 있다.

(4) Universal Approximation Theorem 에 의하면, 입력값에 적절한 상수와 벡터, 그리고 비선형 함수를 이용해 연산을 진행하면 어떠한 수학적 함수 𝑭로도 근사시킬 수 있다.

(5) 최대-최소 정규화는 Z-점수 정규화에 비해 Outlier 의 영향을 적게 받는다.

(6) 전체 데이터 개수가 1200개, Batch Size가 120 인 경우, 1 epoch는 100 iteration 으로 이루어져 있다.

<br>



## Solution 2.

[정답] 3개

(2), (5), (6)이 잘못된 설명

(2) : Sigmoid 함수에서 Gradient Vanishing 이란 역전파가 진행될수록 기울기 값이 **0으로 수렴**하게 되는 문제로, **Leaky ReLU** 함수를 사용해 해결할 수 있다.

(5) : 최대-최소 정규화는 Z-점수 정규화에 비해 Outlier 의 영향을 **많이** 받는다.

(6) 전체 데이터 개수가 1200개, Batch Size가 120 인 경우, 1 epoch는 **10** iteration 으로 이루어져 있다.

<br>

## Problem 3.

아래 설명 중 빈칸 (a), (b), (c), (d), (e) 에 들어갈 수의 합을 구하시오.

---

Sigmoid 함수의 경우 치역이 $(0, a)$ 이며, 중간값은 0.5 이다. 그러나, Sigmoid 함수는 Gradient Vanishing 이라는 문제점을 가지고 있어 실제 DL에서는 잘 사용되지 않는다. 

Sigmoid 함수를 직접 미분해 보자. Sigmoid 함수를 수식으로 나타내면 $\varphi(x) = \frac{1}{1+ e^{-x}}$ 이고, $\varphi'(x) = \varphi(x) (1 - \varphi(x))$ 이다. 실제 숫자를 대입해 계산해 보자. $x=- \ln 3$ 에서 Sigmoid 함수의 미분값은 $\frac{b}{c}$ (단, $b$와 $c$는 서로소인 자연수) 이다.

tanh 함수의 경우 치역이 $(d, 1)$ 이다. 중간값이 0이므로 Sigmoid 함수의 문제점 중 일부를 해결하지만, 역시 Gradient Vanishing 이라는 문제점을 가진다.

ReLU 함수를 수식으로 나타내면 $\varphi(x) = \max (e, x)$ 이다. ReLU 함수의 경우 Dying ReLU 라는 문제를 가지며, 이는 Leaky ReLU 함수를 사용해 해결할 수 있다.

---



## Solution 3.

[정답]

$$a = 1, \ \frac{b}{c} = \frac{3}{16}, \ d = -1, \ e = 0$$

<br>

## Problem 4.

아래 표에서 $(a) + (b)$ 의 값을 구하시오. 

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/kbeFakqf3P1sWQfZ-fNVQ8pQTnNTR-Vndph90MIYpDGu6iLf73UBCu8QxxpN4eJIr14NPKx16dBPij-Kc6KSF8NYZVrmF1L_8kN-jUaEShXLB_xz2PrrOLD8tuFbi7EO.png" width="550"></p>



## Solution 4.

[정답] 0.35

$$(a) + (b) = 1 - (0.15+0.20+0.30)=0.35$$

<br>

## Problem 5.

딥러닝 모델의 Hyperparameter 에 해당하는 것을 모두 고르시오.

- [ ] Learning rate

- [ ] 모델의 가중치

- [ ] 오차함수 (Loss Function)

- [ ] 가중치의 편미분 계수

- [ ] Batch Size


## Solution 5.

- [x] Learning Rate

- [ ] 모델의 가중치

- [x] 오차함수 (Loss Function)

- [ ] 가중치의 편미분 계수

- [x] Batch Size

<br>

## Problem 6.

아래와 같은 계산 그래프의 역전파 과정을 생각해보자. 검정색 화살표는 순전파 과정을, 빨간색 화살표는 역전파 과정을 나타낸다. 

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/ceP2BxS74OXYCRbF_r5K-uzdyOlvA4ufiFar5tZgmAGpvrqGD7R60YyGsjb35stWr-d_PY4e1o9f9Z9tXML5oC43GeKZPLU37iUyYrAu7XLANXgUchyDrB26m_fOEh5G.png" width="790"></p>

$(a)+(b)+(c)+(d)+(e)$의 값을 구하시오. 단, 

$$f(x)=\text{ReLU}(x), \ g(x)=2 \ln{x}, \ h(x) = \text{Sigmoid}(x)$$ 

이다.


## Solution 6.

[정답] $(a) = 0.16, \ (b) = 0.32, \ (c) = 0.16, \ (d) = 0.32, \ (e) = 0.64$


우선, $y =\ln 4, \ z = \frac{1}{1+0.25} = 0.8$ 이다. $\frac{d h(x)}{dx} = h(x)(1 - h(x))$ 이므로, 

$$(a) = 0.8 \times (1 - 0.8) = 0.16$$

이다. 

여기서 $\frac{d g(x)}{d x} = \frac{2}{x}$ 임을 이용하자. 곱셈 노드의 역전파 과정으로부터,

$$(b) = (a) \times \frac{2}{2} \times 2 = 0.32$$

이다. 

$f(x)$ 는 ReLU 함수로, 입력값이 양수이므로 $f(x)=x$ 이고, 따라서 $\frac{d f}{d x} = 1$ 이다. 이로부터

$$(c) = (a) \times \frac{2}{2} \times 1 \times 1 = 0.16$$

를 얻을 수 있다.

덧셈 노드의 역전파 과정으로부터, 

$$(d) = (b) = 0.32$$

이고, 곱셈 노드의 역전파 과정으로부터

$$(e) = (b) \times 2 = 0.64$$

임을 알 수 있다.

<br>

## Problem 7.

아래와 같은 계산 그래프의 역전파 과정을 생각해보자. 검정색 화살표는 순전파 과정을, 빨간색 화살표는 역전파 과정을 나타낸다. 

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/lsnusFw5uU4MyT6jCTtsksfO-zcfOgfwH1Qg566fED14-9hrkqUE_HyWHWH0i89TCUTvhEvZJxYHFyeAI2ekZXptxZbyTbMIYpe3dUXdS-eZBZmxw2hnw_iShutBnvxR.png" width="720"></p>

$(a)$, $(b)$, $(c)$ 각각의 값을 구하시오. 단, 

$$f(x)=\text{Softmax}(\textbf{x})$$ 

$$g(x)=\text{CrossEntropy}(\textbf{x}, \ [0.2, \ 0.1, \ 0.3, \ 0.15, \ 0.25])$$ 


이다.


## Solution 7.

[정답] $(a) = -0.775, \ (b) = -\frac{1.1}{15.5}, \ (c) = -\frac{0.65}{15.5} \ln{⁡2}$



우선 Softmax 함수와 Cross-Entropy 함수의 역전파 과정은 다음 그림과 같이 나타낼 수 있다.

> Figure Source : 사이토 고키, Deep Learning from Scratch, 한빛미디어 (2017)
    <p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile29.uf.tistory.com%2Fimage%2F99EBF5395B98F7792B42CE" width="600"></p>

위 그림에 나타난 방법을 이용해 역전파 계수를 구할 수 있다. 우선, $(a) = -\frac{0.25}{x[4]}$ 이고, $x[4] = \frac{5}{2+1.5+4+3+5} = \frac{1}{3.1}$ 이다. 


$$\therefore (a) = -0.775$$

이다. $(b)$ 의 경우, $(b) = x[0] - 0.2$ 이므로

$$\therefore (b) = -\frac{1.1}{15.5}$$

이다. 곱셈 노드와 덧셈 노드의 역전파 과정으로부터, $(c) = \ln{2} \times (x[2] - 0.3)$ 이므로

$$\therefore (c) = -\frac{0.65}{15.5} \ln{⁡2}$$

<br>

## Problem 8.

아래와 같은 Affine Layer의 역전파 과정을 생각해보자. 검정색 화살표는 순전파 과정을, 빨간색 화살표는 역전파 과정을 나타낸다. 

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/S6c9Ql7RnKWfPCvyME08ym-HsUjkICN0Nmu8FCXFakGYNgUHsUv1B_myqYT1WDH8CVPhjJaHYfwnuM8DmQQjik50cxMcnHYdsCYRVViWZIwPrglGG3KCXknMAfh59c6i.png" width="680"></p>

$(a), (b), (c), (d), (e)$ 각각의 값을 $\frac{\partial L}{\partial \textbf{y}}$ 를 이용해 나타내시오.


## Solution 8.

[정답]

$$\begin{align} & (a) = {\textbf X_1}^{\top} \cdot \frac{\partial L}{\partial \textbf y} \cdot {\textbf H}^{\top} \\\ & (b) = \frac{\partial L}{\partial \textbf y} \cdot {\textbf H}^{\top} \cdot {\textbf W_2}^{\top} \\\ & (c) = \frac{\partial L}{\partial \textbf y} \cdot {\textbf H}^{\top} \\\ & (d) = (\textbf X_1 \cdot \textbf W_1 + \textbf X_2 \cdot \textbf W_2)^{\top} \cdot \frac{\partial L}{\partial \textbf y} \\\ & (e) = \frac{\partial L}{\partial \textbf y} \\\  \end{align}$$


