
# [ML/DL] Quiz : Gradient Descent

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)

## Problem 1.

아래 내용 중 **올바른 문장**의 개수는 몇 개인가?

(1) 경사하강법에서 학습률 (Learning Rate)은 하이퍼파라미터이다.

(2) 경사하강법의 종료 조건으로 Thresholding을 사용할 수 있다. 

(3) 업데이트 할 모델 파라미터를 $w$라고 하자. 학습률을 $\delta$, 손실함수를 $L$이라고 할 때 경사하강법의 과정을 수식으로 나타내면 (단, $\delta > 0$)

$$w_{new} = w_{old} + \delta \cdot \frac{\partial L}{\partial w} \left |\right. _{w = w _ {old} }$$

이다. 

(4) 손실함수가 일변수함수일 때, 현재 가중치 값에서 손실함수 미분값의 부호가 양수이면, 가중치를 더 작은 값으로 변화시켜야 한다.

(5) 손실함수가 이변수함수일 때, 현재 가중치 값에서 손실함수 미분값이 0이라면 현재 가중치 값이 손실함수를 최소로 하는 가중치이므로 경사하강법을 종료할 수 있다. 

<br>

## Problem 2.

$f(x)=x^3$, $\delta=0.1$ 일 때 경사하강법을 통해 $x$좌표의 값을 변화시키는 상황을 생각해보자. 초기 위치 $x = x_0 = 1.0$ 이고, 경사하강법을 $n$회 시행한 후 $x$좌표의 값을 $x_n$ 이라고 하자. 예를 들어,

$$x_1 = x_0 - \delta \cdot \frac{d f(x)}{dx} \left |\right._{x = x_0} = 1.0-0.1\times 3\times 1.0^2 = 0.7$$

이다. $x_3$의 값을 구하시오. 


<br>


## Problem 3.

$f(x)=2xy+3y^3+z^2$, $\delta=0.1$ 일 때 경사하강법을 통해 $(x, y, z)$의 값을 변화시키는 상황을 생각해보자. 초기 위치 $(x, y, z)=(x_0, y_0, z_0)=(1, 1, 1)$ 이고, 경사하강법을 $n$회 시행한 후 $(x, y, z)$의 값을 $(x_n, y_n, z_n)$ 이라고 하자. 예를 들어,

$$(x_1, y_1, z_1) = (0.8, -0.1, 0.8)$$

이다. $x_1+y_2+z_3$의 값을 구하시오. 

<br>


## Problem 4.


아래 코드는 이변수함수

$$f(x, y) = ax^2 + 7y^b$$

에 대해 경사하강법을 시행하는 코드이다. (단, $a$와 $b$는 상수)

```python
import numpy as np

def descent_down_2d_parabola(w_start, learning_rate, num_steps):
    xy_values = [w_start]
    for _ in range(num_steps):
        xy_old = xy_values[-1]
        xy_new = xy_old - learning_rate * (np.array([7., 14.]) * xy_old)
        xy_values.append(xy_new)
    return np.array(xy_values)
```

$a + b = \frac{p}{q}$ 라고 하자. 단, $p$와 $q$는 1 이상의 서로소인 자연수). $p+q$의 값을 구하시오.


<br>

## Problem 5.

아래 코드는 사변수함수

$$f(x, y, z, w) = k_1 xw + k_2 y^2 + k_3 z^{k_4} + k_5 yz + k_6 yw$$

에 대해 경사하강법을 시행하는 코드이다. (단, $k_i$ $(1 \leq i \leq 6)$는 상수이며 $k_4$ 는 2 이상의 정수)

```python
import numpy as np

def descent_down_4d(w_start, learning_rate, num_steps):
    xyzw_values = [w_start]
    for _ in range(num_steps):
        xyzw_old = xyzw_values[-1]
        xyzw_new = xyzw_old - learning_rate * (np.array([[0., 0.,  0.,  2.],
                                                         [0., 14., 3.,  4.],
                                                         [0., 3.,  10., 0.],
                                                         [2., 4.,  0.,  0.]]) @ xyzw_old)
        xyzw_values.append(xyzw_new)
    return np.array(xyzw_values)
```

코드 내 ‘@’ 는 행렬곱을 의미한다. 예를 들어, 행렬 A와 B를 곱한다면 `A @ B` 로 표기한다. $\sum\limits_{i=1}^6 {k_i}$의 값을 구하시오.



## Solution

Check [ML_quiz_gd_sol.md](https://github.com/frogyunmax/OUTTA_2022AIBootcamp/blob/main/ML_quiz_gd_sol.md)
