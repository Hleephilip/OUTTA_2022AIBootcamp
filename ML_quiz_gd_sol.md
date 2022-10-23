
# [ML/DL] Quiz Solution : Gradient Descent

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

## Solution 1.

[정답] 3개

(1) **True**

(2) **True**

(3) False : $\delta > 0$ 이므로 올바른 수식은 아래와 같다.

$$w_{new} = w_{old} - \delta \cdot \frac{\partial L}{\partial w} \left |\right. _{w = w _ {old} }$$

(4) **True**

(5) False : 안장점 (Saddle Point)에 위치한 상황일 수도 있다.


<br>

## Problem 2.

$f(x)=x^3$, $\delta=0.1$ 일 때 경사하강법을 통해 $x$좌표의 값을 변화시키는 상황을 생각해보자. 초기 위치 $x = x_0 = 1.0$ 이고, 경사하강법을 $n$회 시행한 후 $x$좌표의 값을 $x_n$ 이라고 하자. 예를 들어,

$$x_1 = x_0 - \delta \cdot \frac{d f(x)}{dx} \left |\right._{x = x_0} = 1.0-0.1\times 3\times 1.0^2 = 0.7$$

이다. $x_3$의 값을 구하시오. 

## Solution 2.

[정답] $x_3 = 0.4612573$


$$\begin{align} & x_2 = x_1 - \delta \cdot \frac{df(x)}{dx} \left |\right. _{x = x _ {1}} = 0.7 - 0.1 \times 3 \times 0.7^2 = 0.553 \\\ & x_3 = x_2 - \delta \cdot \frac{df(x)}{dx} \left |\right. _{x = x _ {2}} = 0.553 - 0.1 \times 3 \times 0.553^2 = 0.4612573\end{align}$$


<br>



## Problem 3.

$f(x)=2xy+3y^3+z^2$, $\delta=0.1$ 일 때 경사하강법을 통해 $(x, y, z)$의 값을 변화시키는 상황을 생각해보자. 초기 위치 $(x, y, z)=(x_0, y_0, z_0)=(1, 1, 1)$ 이고, 경사하강법을 $n$회 시행한 후 $(x, y, z)$의 값을 $(x_n, y_n, z_n)$ 이라고 하자. 예를 들어,

$$(x_1, y_1, z_1) = (0.8, -0.1, 0.8)$$

이다. $x_1+y_2+z_3$의 값을 구하시오. 

## Solution 3.

[정답] $1.043$

$$\begin{align}& \frac{\partial f(x, y, z)}{\partial x} = 2y \\\ & \frac{\partial f(x, y, z)}{\partial y} = 2x+9y^2 \\\ & \frac{\partial f(x, y, z)}{\partial z} = 2z \end{align}$$


이므로, $y_2, z_3$ 의 값을 계산하면 아래와 같다.


$$\begin{align} & x_2 = x_1 - \delta \cdot \frac{\partial f(x, y, z)}{\partial x} \left | \right. _{x = x_1} = 0.8-0.1 \times (-0.2)=0.82 \\\ & y_2 = y_1 - \delta \cdot \frac{\partial f(x, y, z)}{\partial y} \left | \right. _{y = y_1} = -0.1 - 0.1 \times (1.6+0.09) = -0.269 \\\ & z_2 = z_1 - \delta \cdot \frac{\partial f(x, y, z)}{\partial z} \left | \right. _{z = z_1} = 0.8 - 0.1 \times (2 \times 0.8) = 0.64 \\\ & z_3 = z_2 - \delta \cdot \frac{\partial f(x, y, z)}{\partial z} \left | \right. _{z = z_2} = 0.64 - 0.1 \times (2 \times 0.64) = 0.512 \end{align}$$
 

<br>

$$\therefore x_ 1 +y_2 + z_3 = 0.8  -0.269+0.512=1.043$$


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


## Solution 4.

[정답] $13$

$a = \frac{7}{2}$, $b=2$ 이므로 $a + b = \frac{11}{2}$

$\therefore p + q = 11+2 = 13$


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


## Solution 5.

[정답] 23


우선 $f(x, y, z, w)$ 의 편미분계수를 구해 보자.

$$\begin{align} & \frac{\partial f(x, y, z, w)}{\partial x} = k_1 w
\\\ & \frac{\partial f(x, y, z, w)}{\partial y} = 2 k_2 y + k_5 z + k_6 w \\\ & \frac{\partial f(x, y, z, w)}{\partial z} = k_3 k_4 z^{k_4 - 1} + k_5 y \\\ & \frac{\partial f(x, y, z, w)}{\partial w} = k_1 x + k_6 y \end{align}$$

가 성립한다. 이 때, 경사하강법의 과정이

```python
xyzw_new = xyzw_old - learning_rate * (np.array([[0., 0.,  0.,  2.],
                                                 [0., 14., 3.,  4.],
                                                 [0., 3.,  10., 0.],
                                                 [2., 4.,  0.,  0.]]) @ xyzw_old)
```

처럼 나타나기 위해서는 $k_4 - 1 = 1$ 이 성립해야 한다. 따라서

$$k_4 = 2$$ 

이고, 이를 바탕으로 경사하강법의 과정을 행렬곱을 이용해 나타내면

$$\begin{pmatrix} x_{new} \\\ y_{new} \\\ z_{new} \\\ w_ {new} \end{pmatrix} = \begin{pmatrix} x_{old} \\\ y_{old} \\\ z_{old} \\\ w_ {old} \end{pmatrix} - \begin{pmatrix} 0 & 0 & 0 & k_1 \\\ 0 & 2k_2 & k_5 & k_6 \\\ 0 & k_5 & k_3 k_4  & 0 \\\ k_1 & k_6 & 0 & 0 \end{pmatrix} \begin{pmatrix} x_{old} \\\ y_{old} \\\ z_{old} \\\ w_ {old} \end{pmatrix}$$

이다. 따라서,

$$k_1 = 2, k_2 = 7, k_3 = 5$$

$$k_5 = 3, k_6 = 4$$

이다.

$$\therefore \sum\limits_{i=1}^6 {k_i} = 23$$
