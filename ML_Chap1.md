# [ML/DL] 1장. 머신러닝을 위한 기초 수학

이 자료는 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》 를 바탕으로 제작되었습니다. 

Remnote 자료의 경우 [링크](https://www.remnote.com/a/outta/63525bc8f90f29c65d084baf)를 통해 확인하실 수 있습니다.

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)



## 1. $\mathbb{R}^n$  공간과 함수
**1.1. $\mathbb{R}^n$ 공간과 벡터 ( $n$ 차원 유클리드 공간)**

- 곱집합 $\Leftrightarrow$ $X \times Y=\left\lbrace (x, y): x \in X, y \in Y \rbrace\right.$ 

- $\mathbb{R}^n$ $\Leftrightarrow$ 실수로 구성된 $n$ 차원 벡터의 집합, $\left\lbrace (x_1, x_2, \cdots , x_n) : x_1, x_2, \dots x_n \in \mathbb{R} \rbrace\right.$ 
- **Vector Space** $\Leftrightarrow$ 원소들 간의 합과 상수배가 정의된 공간
- 벡터 연산
    - 벡터의 경우 **볼드체**로 표기한다.

    - 벡터의 덧셈 : $\textbf{x} + \textbf y = (x_1 + y_1, x_2+y_2, \cdots , x_n + y_n )$ 

    - 벡터의 상수배 : $c \textbf x = (cx_1, cx_2, \cdots, cx_n)$ 

    - 벡터의 뺄셈 : $\textbf x  - \textbf y = \textbf x + (-1) \textbf y$ 

    - 벡터의 나란함 : 두 벡터 $\textbf x, \textbf y$가 나란 $\Leftrightarrow$ $\exists s, t \in \mathbb{R}$ $s.t.$ $s\textbf x = t \textbf y$ 

    - 내적과 노름 
        - (Norm) $||\textbf x|| = \sqrt{\textbf x \cdot }\textbf x$ 

        - (Inner product) $\textbf x \cdot \textbf y = x_1y_1 + x_2y_2 + \cdots x_ny_n = \sum\limits_{i=1} ^{n} {x_i y_i}$ 

        - (Inner product with angle) $\textbf x \cdot \textbf y = ||\textbf x|| ||\textbf y|| \cdot \cos\theta$ 

- 내적과 노름 사이의 관계에 관한 부등식
    - 코시-슈바르츠 부등식
    
        $$(\textbf x \cdot \textbf y)^2 \leq ||\textbf x||^2 ||\textbf y||^2$$

        - 등호 성립 조건 $\Leftrightarrow$ $\textbf x // \textbf y$ 

        - Proof) $\textbf{y}$가 영벡터 일 때는 임의의 $\textbf{x}$와 나란하고, 등호가 성립한다.
        $\textbf{y}$가 영벡터가 아니라 가정하자. 그러면 임의의 실수 $t \in \mathbb{R}$에 대해 $|| \textbf{x}+t \textbf{y} || ^2 \geq 0$ 가 성립하므로, 다음 이차식 $$||\textbf{x} + t\textbf{y} || ^2 = || \textbf{y}||^2 t^2 + 2(\textbf{x} \cdot \textbf{y})t + || \textbf{x}||^2$$ 
        의 판별식은 0 이하이다. 즉, $$D/4 = (\textbf{x} \cdot \textbf{y})^2 - ||\textbf{x} ||^2 ||\textbf{y}||^2 \leq 0$$
                이 성립한다. 이 때, 등호가 성립하기 위해서는 이차식의 값이 0이 되어야 한다. Norm 의 성질에 의하여, $$|| \textbf{x} + t\textbf{y} || ^2 = 0 \Leftrightarrow \textbf{x} + t\textbf{y} = \textbf{0}$$ 
        가 성립하여야 한다. $\textbf{y}$가 영벡터가 아니고, $\textbf{x}$와 $\textbf{y}$가 나란하면 위의 식을 만족시키는 $t$가 유일하게 존재하므로 등호가 성립한다. <br></br>
    
    - 삼각 부등식
        $$||\textbf x + \textbf y|| \leq ||\textbf x|| + ||\textbf y||$$


**1.2. 다변수함수와 다변수 벡터함수**
- 함수 :  두 집합 사이의 대응관계

- 다변수 함수
    - $n-$변수 함수 : $f : U \rightarrow \mathbb{R}$ $(U \in \mathbb{R}^n)$ 

    - 내적 역시 $\mathbb{R}^n$ 에서 정의된 다변수함수

- 1절 연습문제 풀기

## 2. 행렬과 선형사상
**2.1. 일차함수**
- $f(x_1, x_2, \cdots , x_n) = a_1 x_1 + a_2 x_2 + \cdots a_n x_n +b$ 

- 벡터로 표현된 $\mathbb{R}^n$ 의 일차함수 

    - $f(\textbf{x})=\textbf{a} \cdot \textbf{x} + b$ (단, $\textbf{a} = (a_1, a_2, \cdots , a_n), \textbf{x} = (x_1, x_2, \cdots, x_n)$ ) 

    - $a_i$ : $x_i$ 방향 기울기
- 함수의 덧셈과 스칼라 곱 
    - $(f+g)(x)=f(x)+g(x)$ 

      $(cf)(x)=c\cdot f(x)$ $(c \in \mathbb{R})$ 

**2.2. 선형사상과 행렬**
- 선형사상 : $L(\textbf x + c\textbf y) = L(\textbf x) + cL(\textbf y)$ 을 만족하는 사상 $L$ (단, $c \in \mathbb{R}$)

    - $L : \mathbb{R}^n \rightarrow \mathbb{R}$ 의 경우, $L(\textbf x) = \textbf a \cdot \textbf x$ 로 표현 가능 $(\textbf{a}, \textbf{x} \in \mathbb{R}^n)$

    - **"사상은 변환이다"**

- 다변수 벡터함수에서의 선형사상 $(L : \mathbb{R}^n \rightarrow \mathbb{R}^m )$ 

    - $L(\textbf x) = \textbf a_1 x_1 + \textbf a_2 x_2 + \cdots \textbf a_n x_n$ , 각 변수의 차원은? 

        - $\textbf a_i \in \mathbb{R}^m$  

        - $x_i \in \mathbb{R}$ 

    - $\textbf a_i = L(\textbf e_i )$ 
- 행렬과 선형사상 : $L(\textbf x) = A\textbf x$ 

    - $L(\textbf x) = \textbf a_1 x_1 + \textbf a_2 x_2 + \cdots \textbf a_n x_n$  에 대해,

        $$ A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\\ a_{21} & a_{22} & \cdots & a_{2n} \\\ \vdots & \vdots & \ddots & \vdots \\\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} = \begin{bmatrix} | & | &  & | \\\ \textbf{a} _1 & \textbf{a} _2 & \cdots & \textbf{a} _n \\\ | & | &  & | \end{bmatrix}$$

        - $A = [a_{ij}]_{m \times n}$

        - $\textbf x \in \mathbb{R}^{n \times 1} , \in M_{n \times 1}$ (열벡터)


    - 행렬 연산

        - 합

            $$ A+B = [a_{ij} + b_{ij}] = \begin{bmatrix} a_{11} + b_{11} &  a_{12} + b_{12}& \cdots & a_{1n} + b_{1n} \\\ a_{21} + b_{21} &  a_{22}+ b_{22} & \cdots &  a_{2n}+ b_{2n} \\\ \vdots & \vdots & \ddots & \vdots \\\ a_{m1} + b_{m1}&  a_{m2}+b_{m2} & \cdots &  a_{mn}+b_{mn} \\\ \end{bmatrix} $$
            

        - 상수배

            $$ cA = [ca_{ij}] = \begin{bmatrix} ca_{11} & ca_{12} & \cdots & ca_{1n} \\\ ca_{21} & ca_{22} & \cdots & ca_{2n} \\\ \vdots & \vdots & \ddots & \vdots \\\ ca_{m1} & ca_{m2} & \cdots & ca_{mn} \\\ \end{bmatrix}$$


        - 행렬곱

            $$ AB_{ij} = a_{i1} b_{1j} + a_{i2} b_{2j} + \cdots + a_{im} b_{mj} = [A]_{i} \cdot [B]^{j} $$  

        - 일반적으로, 교환법칙이 성립하지 않음

<br>
<br>


**2.3. 함수의 합성과 행렬의 곱**

- $L_1 : \mathbb{R}^l \rightarrow \mathbb{R}^r$, $L_2 : \mathbb{R}^n \rightarrow \mathbb{R}^m$ 일 때 $(L_1 \circ L_2)(\textbf x)$ 은? (대응 행렬은 각각 $A, B$) 

    -  $m=l$

    - $(L_1 \circ L_2)(\textbf x) = (BA) \textbf x$ , $(\forall \textbf x  \in \mathbb{R}^n)$ 

- 전치행렬과 대칭행렬

    - $A \in \mathbb{R}^{m \times n}$ 에 대해, $A_{ij} = a_{ij}$ 의 전치행렬 $A^T$ 는 $A_{ij}^T = a_{ji}$ 를 만족

    - 대칭행렬 : $A = A^T$ 인 행렬 $A$

- 내적과 행렬곱

    - **열벡터를 행렬로 바라볼 수 있다!**

        $$ \textbf{x} = 
        \begin{bmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_n
        \end{bmatrix} = \begin{pmatrix} x_1, \; x_2, \; \cdots, \; x_n \end{pmatrix}^T$$

    - $\textbf x \cdot \textbf y = \textbf x ^T \textbf y$

        Proof)

        $$ \textbf{x} \cdot \textbf{y} = x_1y_1 + x_2y_2 + \cdots + x_ny_n 
        = \begin{bmatrix} x_1 & x_2 & \cdots  & x_n \end{bmatrix}
        \begin{bmatrix}
            y_1 \\
            y_2 \\
            \vdots \\
            y_n
        \end{bmatrix}$$


    - **선형사상이 행렬인 이유**를 설명

- 2절 연습문제 풀기

## 3. 미분 

### **3.1. 미분과 선형사상**
    
**3.1.1. 일변수함수의 미분**

- 미분계수 
    $$f'(x_0) = \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0}$$

- 미분 가능 함수 : 정의역의 모든 점에서 미분계수가 존재하는 함수

- 도함수 
    $$f'(x) = \lim_{h \rightarrow 0}\frac{f(x+h)-f(x)}{h}$$

- 미분계수의 기하하적 의미 : 접선의 방정식을 의미

    - $y = f'(x_0)(x-x_0) + f(x_0)$ 
    
    - Proof) 

    $$ \lim_{x \rightarrow x_0} \frac{f(x) - [f'(x_0)(x-x_0) + f(x_0)]}{x-x_0} = \lim_{x \rightarrow x_0} \left( \frac{f(x) - f(x_0)}{x-x_0}  - f'(x_0) \right) = f'(x_0)-f'(x_0) = 0$$

- '근사' 로서의 미분 : 그 함수와 가장 가까운 일차함수를 찾게 해주는 도구


**3.1.2. 다변수함수의 미분**


- 미분계수 $f'(\textbf x_0)$ $(=Df(\textbf x_0))$ :

    $$\lim_{\textbf{x} \rightarrow \textbf{x}_0} {\frac{|f(\textbf{x}) - (\textbf{a}^T \textbf{x} + \textbf{b})|}{|| \textbf{x} - \textbf{x}_0 ||}} = 0$$
    
    을 만족시키는 벡터 $\textbf a^T$ 

- 미분가능 조건 : 벡터 $\textbf a, \textbf b$  가 존재


**3.1.3. 다변수 벡터함수의 미분**
- 미분계수 $f'(\textbf x_0)$ $(=Df(\textbf x_0))$ : 

    $$\lim_{\textbf{x} \rightarrow \textbf{x}_0} {\frac{|f(\textbf{x}) - (A \textbf{x} + \textbf{b})|}{||\textbf{x}-\textbf{x}_0||}} = 0$$

    을 만족시키는 행렬 $A$ 

- 미분가능 조건 : 행렬 $A$, 벡터 $\textbf b$  가 존재

**3.1.4. 함수의 연속**

- 함수 $f(\textbf x)$가 $\textbf x_0$에서 연속 :

    $$\lim_{\textbf x \rightarrow \textbf x_0}f(\textbf x) = f(\textbf x_0)$$

### 3.2. 다변수함수의 미분

**3.2.1. 편미분 (Partial Derivative)**

- 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에 대해, 점 $\textbf x_0 = (x_1, x_2, \cdots , x_n)^T$ 에서의 $i$번째 편미분계수

    $$\frac{\partial f}{\partial x_i} (\textbf x_0) = \lim_{x \rightarrow x_{i}} {\frac{f(x_1, \cdots, x, \cdots, x_n) - f(x_1, \cdots, x_i, \cdots, x_n)}{x-x_i} = \lim_{h \rightarrow 0}{\frac{f(\textbf{x}_0 + h\textbf{e}_i) - f(\textbf{x}_0)}{h}}}$$

- 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에 대해, $i$번째 편도함수

    $$ \frac{\partial f}{\partial x_i} (x_1, \cdots, x_n) = \lim_{x \rightarrow x_i} {\frac{f(x_1, \cdots, x_i + h, \cdots, x_n) - f(x_1, \cdots, x_i, \cdots, x_n)}{h}}$$



- 편도함수의 기호 : $D_i f, f_i, \frac{\partial f}{ \partial x_i}$ 


**3.2.2. 그래디언트 벡터 (Gradient Vector)**
- 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에 대해, 점 $\textbf x_0$ 에서의 그래디언트 벡터 $(\nabla f(\textbf x_0))$ 

    $$ \nabla f(\textbf{x}) = \left ( \frac{\partial f}{\partial x_1}(\textbf{x}), \frac{\partial f}{\partial x_2}(\textbf{x}), \;\cdots,\;
    \frac{\partial f}{\partial x_n}(\textbf{x}) \right ) ^T$$

    **Theorem 2**

    ---

    다변수함수 $f : \mathbb{R}^n \rightarrow \mathbb{R}$ 가 $\textbf x=\textbf x_0$ 에서 미분가능하다면, Gradient 는 미분가능한 다변수함수 $f$의 미분계수와 같다. $$f'(\textbf{x}_0) = \nabla f(\textbf{x}_0)^T$$

<br>

- **일급함수** (**$C^1$** 함수) : 1계 미분가능하고, 각 편도함수가 모두 연속인 함수 <br>


    **Theorem 3**

    --- 

    일급함수 **$f : \mathbb{R}^n \rightarrow \mathbb{R}$** 는 미분가능하고, Theorem 2가 성립한다.

</br>

- 야코비 행렬 $\left(J_f (\textbf x ) \right)$
    
    $$J_f(\textbf{x}) =  \begin{vmatrix} - & \nabla f_1(\textbf{x})^T & - \\\ - 
    & \nabla f_2(\textbf{x})^T & - \\\ & \vdots & \\\ - & \nabla f_m(\textbf{x})^T & - \end{vmatrix} = \left[ \frac{\partial f_i}{\partial x_j} \right]_{m \times n}$$


    **Theorem 4**

    --- 

    다변수 벡터함수 $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ 가 $\textbf x = \textbf x_0$ 에서 미분가능하면, 야코비 행렬이 존재하고, $$f'(\textbf x_0) = J_f (\textbf x_0 )$$

<br>


### 3.3. 연쇄법칙 (Chain Rule) 
- $$\frac{d(g \circ f)}{dx} = \sum_{i=1} ^ n {\frac{\partial g}{\partial x_i} }\frac{df_i}{dx}$$

    **Theorem 5 (Chain Rule)**
    
    --- 
    
    함수 $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ 가 $\textbf{x}_0$ 에서 미분가능하고, 함수 $g : \mathbb{R}^m \rightarrow \mathbb{R}^r$ 가 $f(\textbf{x}_0)$ 에서 미분가능할 때, 그 합성 $g \circ f$ 는 $\textbf{x}_0$ 에서 미분 가능하고 다음이 성립한다. $$(g \circ f)'(\textbf{x}_0) = g'(f(\textbf{x}_0))f'(\textbf{x}_0) = J_g (f(\textbf{x}_0))J_f(\textbf{x}_0)$$

<br>

### 3.4. 다변수함수의 최적화 
**3.4.0. 최적화 (Optimization)** : 함수의 값을 최소화 혹은 최대화하는 작업

**3.4.1. 극대와 극소** 
- 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$ 가 $\textbf{x} = \textbf{x}_0$ 에서 극대 $\Leftrightarrow$ $^\exists \epsilon > 0$ $s.t.$ $||\textbf{x} - \textbf{x}_0|| < \epsilon \Rightarrow f(\textbf{x}_0) \geq f(\textbf{x})$ 

    **Theorem 6 (임계점 정리)**

    --- 

    미분가능한 함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$ 가 점 $\textbf{x}_0$ 에서 극값을 가지면 $$\nabla f(\textbf {x}_0) = \textbf 0$$

<br>

- 임계점정리의 역은 성립하지 않음 : 안장점이 존재할 수 있기 때문

**3.4.2. 방향미분계수**

- 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에서 단위 벡터 $\textbf{v}$에 대한 $\textbf{x} = \textbf x_0$에서의 $\textbf{v}-$방향 변화율 $(D_{\textbf{v}} f(\textbf x_0))$

    $$ D_\textbf{v} f(\textbf x_0) = \lim_{h \rightarrow 0}{\frac{f(\textbf x_0 +h\textbf{v}) - f(\textbf x_0)}{h}}$$


- $D_{- \textbf{v}}$ 에 대해 아래 식이 성립한다.

    $$D_{- \textbf{v}}f(\textbf{x}_0) = -D_{\textbf{v}}f(\textbf{x}_0)$$


    **Theorem 7**

    --- 

    미분가능한 다변수함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$ 와 임의의 단위 벡터 $\textbf{v}$에 대해, $\textbf {v}-$방향의 변화율이 존재하고 $$D_{\textbf {v}}f(\textbf{x}_0) = \nabla f(\textbf{x}_0)^T \textbf{v}$$

<br>

- 가장 가파른 증가/감소 방향
    - $|D_{\textbf {v}} f(\textbf {x}_0)| \leq||\nabla f(\textbf {x}_0)||$ 

    - 가장 가파른 증가 방향 : $D_{\textbf {v}}f(\textbf {x})$가 가장 큰 $\textbf{v}$ $\Leftrightarrow$ $\nabla f(\textbf{x}_0)/||\nabla f(\textbf{x}_0)||$ 

    - 가장 가파른 감소 방향 : $D_{\textbf {v}}f(\textbf {x})$가 가장 작은 $\textbf {v}$ $\Leftrightarrow$ $- \nabla f(\textbf{x}_0)/||\nabla f(\textbf{x}_0)||$ 



- 3절 연습문제 풀기
<br><br>


## Appendix
> 이 절은 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》 에서 가져왔습니다.

**1. Proof of Thm 2**

<img src="https://remnote-user-data.s3.amazonaws.com/3rCF-Cs59HkAo_QOSTjD86J3KSQ9mRwqFsbVLy2Xw9xqCYwYR97Y68Kbwe_We2S6lkizyNdUvFuo04ZkM2jEntd7dxZeQUVjeUeG28dyaKA1R9FFiF7ZxXDtVvb7T6PK.png" width="600">

<br> 

**2. Proof of Thm 5**

<img src="https://remnote-user-data.s3.amazonaws.com/kJ8fzVo9WyV933ghhYwlvlp8uiMXG2OeV6uIFsj_NhmZA187dOkXmGSlhbUq3tTNegYPDPNb8_5AL5DTizRY60Np9Y9uhcay0jBfj6brsDyGxm-7iRaaIz-naNm1nySh.png" width="620">

<img src="https://remnote-user-data.s3.amazonaws.com/a20WlHnL5e1Q-7j31VNlhm057KeHh3anc4bJuiCGjrLU2E8t937DailSAWcFulrBJVh7XCQe0-s0SpOCjHhO_igZcOgNMVdyp5Jn64CZEPuDKzHOMl751WkxgzChFYoP.png" width="620">

<br>

**3. Proof of Thm 7**

<img src="https://remnote-user-data.s3.amazonaws.com/xWjNc6U6Zf1YH248DIfDgzfWoPAASrXSRq1IjT88cPAvr1PGEi82AHJdzi85jwLjdqVZqKeOr99zTccZnK7F7bKJY3yidBMxfLqysfE6poBr995juOgQzfPO3KkA4P_E.png" width="650">

<br>

## References
>[1] 인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기, OUTTA, 2022<br>
>[2] 미적분학 1+, 김홍종, 서울대학교출판문화원, 2016<br>
>[3] 미적분학 2+, 김홍종, 서울대학교출판문화원, 2016<br>
>[4] 해석개론, 김성기, 김도한, 계승혁, 서울대학교출판문화원, 2011
