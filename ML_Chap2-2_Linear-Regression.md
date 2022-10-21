
# [ML/DL] 2장 - (2). Linear Regression

이 자료는 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》 를 바탕으로 제작되었습니다. 

원본 Remnote 자료의 경우 [링크](https://www.remnote.com/a/-ml-dl-2-2-/63525e35b8761d1e84ad3586)를 통해 확인하실 수 있습니다.

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)



## 2.0. 용어 정의
- 모델 (Model) : 관찰한 데이터로부터 예측과 결정을 얻어내는 수학적 함수

- 모델 파라미터 (Model Parameters): 올바른 예측과 결정을 얻기 위해 조정(tuning)하는 변수들 모델의 함수식을 결정하는 역할을 하는 변수들
- 손실 함수 (Loss function) : 모델의 질을 평가하는 함수

## 2.1. 선형 회귀의 정의
- 선형 회귀 (Linear Regression) : 독립변수와 종속변수에 대해, 두 변수의 관계를 설명할 수 있는 선형 함수를 찾아내는 과정

- 종속 변수 : $X$, 독립 변수 : $Y$

## 2.2. 선형 모델 세우기

- $F(m, b; x)=mx+b$ 
- 모델 파라미터 : $(\text{slope})\; m, \; (\text{intercept})\;b$ 
- 최적의 모델을 위한 m, b의 값을 찾아내는 방법 : **최소제곱법**


## 2.3. 최소제곱법 (MSE)
- 이상치 (Outlier) : 주어진 데이터의 전체적인 경향성에서 크게 벗어난 값
    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/4BbnXRei4wQJjEfe3OUI6VX8E_irW1zt2HaSx8FCThuLvTnqTBwrh0qicQs4cd7eIH5fM60SkGliQswXdpmVw55Ni3uob2Gk4OBPIc9yJcn1bwoh4Uv4e4YfqipaWmTW.png" width="400"></p>

- 최소제곱법을 사용하는 경우, 이상치가 없어야 (적어야) 잘 적용됨.
    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/Mopgwl2FejpdOI-L5evN_DprmAMDo3dUHcupPaW-n4v3GUSXketulRhGDctDaHecpVflq-EhGwLkV69ZXd0yvOaKFvmhCiJMyogIm8HLldkVsRW0LMcxIwRl3SZvqpDH.png" width="400"></p>

- 수학적 모델링

    - Data points 

        - 전체 데이터 개수 $N$개

        - True Data points : $(x_i, y_i ^{(true)}), \; 0 \leq i \leq n - 1$ 

        - Expected Data points : $(x_i, y_i ^{(pred)}), \; y_i^{(pred)} = mx_i + b, \;0 \leq i \leq n - 1$ 

    - 잔차 (Residual)
        - 잔차 : $d_i = (y_i^{(true)} - y_i ^{(pred)})$ 

        - **잔차 제곱합 (RSS)** $$RSS = \sum_{n=0}^{N-1} d_i^2 = \sum_{n=0}^{N-1}(y_i^{(true)} - y_i ^{(pred)})^2$$  
            - Q) 잔차의 합 대신 제곱을 사용하는 이유? 
                
              A) 잔차의 부호가 고려되지 않기 때문. +, - 부호가 섞여 있는 경우 **잔차의 합은 선형 회귀 모델의 오차를 Perfectly represent 하지 못한다.** 

            - Q) 잔차의 절댓값 합 대신 제곱합을 사용하는 이유? 

                A) 절댓값을 사용한 경우 모델 오차로써의 의의는 가지지만, **이를 최적화 하는 과정이 제곱합을 이용했을 때에 비해 복잡하기 때문**. 여러 최적화 기법 중 미분을 사용하는데, 절댓값의 합으로 이루어진 함수를 미분해 최소값을 찾는 과정은 복잡함. 반면 제곱합을 미분해 최솟값을 찾는 과정은 상대적으로 컴퓨터가 연산하기 수월함.


        - 선형 회귀 모델이 학습을 통해 최소화 하고자 하는 값 $=$ **RSS** 
    - 손실 함수 (Loss Function)

        - RSS를 이용해서 정의 $$L(m, b) = \sum_{n=0}^{N-1}(y_i^{(true)} - F(m, b;\; x_n ))^2$$ 

    - 최소제곱법 

        - **손실 함수** $L(m, b)$를 최소화하는 $m^{\star}, b^{\star}$ 구하기

        $$m^{\star}, b^{\star} = \mathrm{argmin} _{m, b \in \mathbb{R}}(L(m, b))$$

- 위험도와 경험적 위험도 최적화
    - Definition of  MSE (Risk, 위험도)  $$\frac{1}{N} \sum_{n=0}^{N-1}(y_n ^{(true)}  - y_n ^{(pred)})^2$$
    - 경험적인 위험도 (Empirical Risk) : 학습용 데이터셋에 대해, 개별 데이터 각각의 **손실의 평균**을 구한 것

        - ex) MSE, |Residual|, Cross-Entropy 

        - **경험적 위험도 최적화** : Train Datasets의 경험적인 위험도를 감소시켜 Optimal Model 을 찾는 과정

<br>

- 과적합 (Overfitting) : Train data 수가 적을 때, Train data에 모델이 과하게 학습되어 Test data에 대한 성능이 감소하는 경우

    - 구조적 위험도 최소화(Structural Risk Minimization) 

        - 기존의 오차 함수 (경험적 위험도)에 **Model의 Complexity에 관한 Penalty 항을 추가**해, 모델의 Performance와 Complexity 간의 균형을 맞추는 학습 방법

        - Overfitting 방지

        - Ex) 가중치 규제(Weight Decay), 배치 정규화(Batch Normalization), 규제(Regularization), etc... 

<br>

## 2.4. $m^{\star}$ 와 $b^{\star}$ 구하기
- Method 1 : 직접 구하기 (Brute-Force)
    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/BC3zZ4AGWaarvp2IAxw8nMf33jN_VZvFamevNDLUe7A97_KmQFe3yMZ-lwqMZzWsozP1HUK-2arB6dRBfb95dzWFHaOa4ZAnMhaRNAv7XrCv-gKGK5Dqrb6UHSwnIAEX.png"></p>


    (0) 원리 : 손실 함수가 최솟값을 가질 때 : 
    
    $$\nabla L(m, b)| _{m=m^{\star}, \;b=b^{\star}} = 0$$

    (1) 손실 함수 $L(m, b)$ 전개

    $$ \mathscr{L}(m, b; (x_n, y_n)^{N-1}_{n=0}) 
    = \sum _{n=0} ^{N-1}{(y_n - (mx_n + b))^2} 
    = \sum _{n=0} ^{N-1}{(m^2 x_n^2 + b^2 + y_n ^2  + 2bmx_n - 2mx_n y_n - 2by_n)}$$
    
    (2) $L(m, b)$을 $m, \; b$에 대해 **편미분** 

    $$ \frac{\partial \mathscr{L} (m, b)}{\partial m} = \sum_{n=0}^{N-1} (2mx_n^2 + 2bx_n - 2x_n y_n ) = 0 $$

    $$ \frac{\partial \mathscr{L} (m, b)}{\partial b} = \sum_{n=0}^{N-1} (2b + 2mx_n - 2y_n) = 0
    $$

    (3) 편미분 식으로부터 $m^{\star}, b^{\star}$ 구하기

    $$m^{\star}=\frac{\sum\limits_{n=0} ^{N-1} {x_{n} y_{n}} - \frac{1}{N} \sum\limits_{n=0} ^{N-1}{x_n} \sum\limits_{n=0} ^{N-1}{y_n}}{\sum\limits_{n=0} ^{N-1} {{x_n}^2} - \frac{1}{N} (\sum\limits_{n=0} ^{N-1}{x_n} )^2}$$

    $$b^{\star} = \bar{y} - m^{\star} \bar{x}$$ 

    - 이렇게 구한 $m^{\star}$와 $b^{\star}$ 를 '**최소제곱추정량**' 이라 한다. 

    <br>


    (4) 최소제곱법을 사용할 수 없는 경우 
    1. Dataset의 Data point 개수 $\leq$ 1
    2. 모든 Data point가 같은 $x_i$ 값을 가지는 경우
    
    <br> 

    (5) *Additional : Using Matrices*
    > 아래 내용은 서울대학교 데이터사이언스 대학원 MLDL1 Course의 Lecture Notes 를 인용하였습니다.


    Describe Linear Regression Model,
    
    $$\textbf{y} = \textbf{X} \textbf{M} + \textbf{e}$$  

    where
    
    $$\textbf{y} = \begin{pmatrix} y_{1} \\\ y_{2} \\\ \vdots \\\ y_{n} \end{pmatrix}, \textbf{X}= \begin{pmatrix} 1 & x_1 \\\ 1 & x_2 \\\ \vdots &\vdots \\\ 1 & x_n \end{pmatrix}, \textbf{M}= \begin{pmatrix} m \\\ b \end{pmatrix}, \textbf{e} = \begin{pmatrix} \epsilon_1 \\\ \epsilon_2 \end{pmatrix}$$  
    
    Then, 
    
    $$RSS=(\textbf y  - \textbf X \textbf M)^T (\textbf y  - \textbf X \textbf M)$$
    
    Using Partial derivative,
    
    $$\frac{\partial (RSS)}{\partial M^{\star}} = -2 \textbf X^T \textbf y + 2 \textbf X^T \textbf X \textbf M^{\star} = 0$$
    
    Finally, we get ordinary least squares parameter estimates.
    
    $$M^{\star} = \begin{pmatrix} m^{\star} \\\ b^{\star}\end{pmatrix} = (\textbf X^T \textbf X)^{-1} \textbf X^T \textbf y$$ 



- Method 2: Gradient Descent
    - In Chapter 2.3.

