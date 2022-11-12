
# [ML/DL] Quiz : Linear Regression

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)

## Problem 1.

아래 내용 중 올바른 문장의 개수는 몇 개인가?

(1) 최소 제곱법은 Outlier가 많은 데이터셋보다는 Outlier가 적은 데이터셋에 대해 더 좋은 성능을 나타낸다.

(2) 경험적 위험도를 최소화시키면 임의의 데이터에 대해서도 좋은 성능을 나타내는 모델을 얻을 수 있다.

(3) 잔차의 제곱을 최소화 시키는 모델을 선형 회귀 모델로 사용할 수 있다.

(4) 잔차 제곱합 (RSS)와 잔차의 합 모두 선형 회귀 모델의 손실함수로써 사용될 수 있다.

(5) Overfitting 이란 Test data에 모델이 과하게 학습되어 Train data에 대한 성능이 감소하는 현상을 의미한다.


<br>

## Problem 2.

구조적 위험도 최소화(Structural Risk Minimization)의 예시로 올바른 것을 모두 고르시오. 

- [ ] Weight Decay

- [ ] Batch Normalization

- [ ] Learning rate Decay

- [ ] L1 Regularization

- [ ] Pooling

- [ ] Convolution

<br>

## Problem 3.

아래와 같은 Data Point 5개에 대해, 선형 회귀를 적용하고자 한다. 선형 회귀 모델을

$$y = mx + b$$

로 나타낼 때, $(m, b)$ 의 최소제곱추정량을 $(m^{\star}, b^{\star})$ 라고 하자.  $m^{\star} + b^{\star}$의 값을 구하시오.

| $x_i$ | $y_i$ |
|:---:|:---:|
|1.0|1.8|
|2.0|4.1|
|3.0|6.0|
|4.0|8.5|
|5.0|9.8|


<br>

## Problem 4.

아래 코드는 선형 회귀 모델의 최소제곱추정량을 찾는 함수이다.

```python
import numpy as np

def ordinary_least_squares(x, y):
    N = x.size
    m = (np.matmul(x, y) - f(x) * y.sum() / N) / (g(x) - (1 / N) * x.sum() ** c )
    b = y.mean() - m * x.mean()
    return m, b
```

5번째 줄의 $f(x), g(x)$ 는 모두 변수 $x$에 대한 연산을 나타내며, $c$는 상수이다. 변수 $x$ 의 값이

  `x = np.array([1., 5., 2.5, 8., 10.])`

일 때, $x$ 에 대한 $f(x), g(x)$ 값을 각각 $a, b$ 라고 하자. $a+b+c$의 값을 구하시오.

<br>

## Solution

Check [ML_quiz_Linear-Regression_sol.md](https://github.com/frogyunmax/OUTTA_2022AIBootcamp/blob/main/ML_quiz_Linear-Regression_sol.md)
