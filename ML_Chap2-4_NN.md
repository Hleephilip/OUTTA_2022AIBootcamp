# [ML/DL] 2장 - (4). Neural Networks
이 자료는 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》 를 바탕으로 제작되었습니다. 

[Remnote 자료](https://www.remnote.com/a/-ml-dl-2-4-/63539d474d226e6f6b88fc60)와 [Lecture PPT](https://github.com/frogyunmax/OUTTA_2022AIBootcamp/blob/main/ML_Chap2-4_NN_Lecture.pdf)는 링크를 통해 확인하실 수 있습니다.

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)


## 4.1. 신경망 활용의 이점: Universal Approximation Theorem

- **신경망 (Neural Networks, NN)** : 입력(input) 데이터와 출력(output) 데이터 사이의 관계를 하나의 함수 관계로 연결해 표현해주는 인공지능 모델
- 신경망을 이용하면, 특정 조건을 만족하는 함수들에 대해 충분한 근사치를 얻을 수 있음이 증명되었음 : Universal Approximation Thm.

- *Universal Approximation Theorem*
    
    $\varphi, I_m, C(I_m), f$ 를 아래 표와 같이 정의하자.

    | Notation | Meaning |
    |:---:|---|
    | $\varphi$ | Nonconstant, continuous, bounded, and monotonically increasing non-linear function |
    | $I_m$ | $R^m$의 닫힌 유계 부분집합 |
    | $C(I_m)$ | $I_m$ 에서 $\mathbb{R}$ 로 가는 연속함수의 집합 |
    | $f$ | $C(I_m)$ 에 속하면서 $\varphi$ 와 독립인 임의의 함수 |

    위 표의 조건을 만족하는 $\varphi$ 에 대해, 

    $$F(\left\lbrace v_i \rbrace\right., \left\lbrace \textbf w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; \textbf x) = \sum\limits_{i=1}^{N} v_i \varphi(\textbf{x} \cdot \textbf{w}_i + b_i)$$

    

    와 같이 함수 $F$ 를 정의하면, 역시 위 표의 조건을 만족하면서 정의된 임의의 함수 $f$ 에 대해,

    $$^{\forall} \epsilon > 0, ^{\exists} \left\lbrace v_i \rbrace\right., \left\lbrace \textbf{w}_i \rbrace\right., \left\lbrace b_i \rbrace\right. \text{ s.t. } |F(\textbf{x}) - f(\textbf{x})| < \epsilon$$ 

    즉, Universal Approximation Theorem 은 Proper 한 비선형 함수 $\varphi$ 를 이용해, $F(\left\lbrace v_i \rbrace\right., \left\lbrace \textbf{w}_i \rbrace\right., \left\lbrace b_i \rbrace\right.; \textbf{x})$ 가 $C(I_m)$ 에 속하면서 $\varphi$ 와 독립인 임의의 함수 $f$ 에 매우 가까워지게 하는 파라미터 $\left\lbrace v_i \rbrace\right., \left\lbrace \textbf{w}_i \rbrace\right., \left\lbrace b_i \rbrace\right.$를 항상 찾을 수 있음을 의미한다.

    다시 말하자면, $\textbf{x}$ 값에 대해 적절한 선형 및 비선형 연산을 적용한다면, $f(\textbf{x})$ 의 값으로 근사시킬 수 있다. '선형 및 비선형 연산' 은 파라미터와 비선형 함수로 표현할 수 있고, 이 파라미터을 신경망 모델로 표현하고 적절한 값을 찾아내는 것이 모델 학습(Training)의 과정이다.
    
    > Universal Approximation Theorem 에 대한 자세한 내용은 <인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기> 본문을 참고해 주시기 바랍니다.

<br></br>

## 4.2. 신경망의 구조

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/Z2uU83wVrmax1zVov15o81-pkBgPOBboDjy9tR3iHT-2-9Ft4b-UeXcncMdTw2xzEByRucb9GQ7z8kh-vSKw3esM8uTYVX4kLddpa4VV_mXcZ3ZqKUnjQWQujsh0VuvO.png" width="250"></p>


- In Universal Approximation Thm (위 표에 나타난 모든 가정을 만족하는 상태에서),

$$F(\left\lbrace v_i \rbrace\right., \left\lbrace \textbf w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; \textbf x) = \sum\limits_{i=1}^{N} v_i \varphi(\textbf x \cdot \textbf w_i +b_i ) $$

$$(v_i, b_i \in \mathbb{R}, \textbf w_i \in \mathbb{R}^m)$$ 

<br>

### 4.2.0. 신경망과 방향 그래프

- $N=5$일 때 **일변수함수** $F$의 모식도를 방향 그래프 (Directed Graph)로 나타내기

    $$F(\left\lbrace v_i \rbrace\right., \left\lbrace w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; x) = \sum\limits_{i=1}^{5}v_i \varphi( x \cdot  w_i +b_i )$$ 


    - 방향 그래프 구성 요소 : 1) 노드,  2) 간선 (노드를 이어주는 선) - 간선은 방향을 가짐
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/-UzDBy62Aztr5UueetRGUYoDTtwR5yB0IBcZM4x0dnc3k8jNNUwM2z_G8Ehf5BNKpIqded71ri20XiUP2_GmIckexgGPhk9cD2tdGG1K1jbbAIXmEcowWGusvnOKOGk-.png" width="300"></p>
    
    
        그래프의 가중치와 편향은? 

        - 가중치 : $w_i$  (입력값에 곱해주는 값)

        - 편향 : $b_i$  (함수 $\varphi$에서 더해주는 값)

    - 참고) 무방향 그래프 : 간선의 양방향으로 모두 이동할 수 있는 그래프

    - **편향을 가중치로 취급**하는 방법?

        - 모델의 입력값을 $x$와 '1' 두 개로 생각

            $$F(\left\lbrace v_i \rbrace\right., \left\lbrace w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; x) = \sum\limits_{i=1}^{5}v_i \varphi( x \cdot  w_i + 1\cdot b_i )$$ 

            <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/EqgbOKdcQleoU0uH69djlTjbNq6FKVw_t0VciYlwpTqWPYOf93viEpWY7m7tIieBzyamPv8NNWTvN9ZA6TWg-woH5biMQJctWUn5eBMS24LMVHkBDFD9o7I9dRpfV2Gc.png" width="400"></p>
        
- $N=5$일 때 **이변수함수** $F$의 모식도를 방향 그래프 (Directed Graph)로 나타내기 

    $$F(\left\lbrace v_i \rbrace\right., \left\lbrace \textbf w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; \textbf x) = \sum\limits_{i=1}^{5}v_i \varphi(\textbf x \cdot \textbf w_i +b_i )$$ 


    $$\textbf{x} = (x_1, x_2)^T, \; \textbf{w}_i = (w_{i_1}, w_{i_2})^T$$ 

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/GxcmQshz2Ze-J41pzVzri6vA6JVU3She7VOtjtLOISdJMucZaOQ299Lu-3emDDQAly1Xni1YxgizjZvIeSWVkCkNM7S-l-BN_Z2Ko8ZAdzsf1lF9esJx2_ade4LA3_KN.png" width="380"></p>



    - 편향을 가중치처럼 본다면?

        $$F(\left\lbrace v_i \rbrace\right., \left\lbrace \textbf w_i \rbrace\right., \left\lbrace b_i \rbrace\right. ; \textbf x) = \sum\limits_{i=1}^{5}v_i \varphi(\textbf x \cdot \textbf w_i + 1 \cdot b_i )$$ 


        $$\textbf{x} = (x_1, x_2)^T, \textbf w_i = (w_{i_1}, w_{i_2})^T$$
        
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/LYa2O3GpwH_QfUmfVIYqmK9YKdR97nIRj3KE_Vmsw28oQgsuadfsKNg_vkXgeeQWDCWTVX7Osc5a2Gpe4aLHy3M2u0iobEevMDXt8b2l6yriHQU3hvjQ1bSnAL4UZd6Y.png" width="380"></p>

<br>

### 4.2.1. 신경망의 기본 구조

- 노드와 간선으로 이루어진, 방향이 정해져 있는 그래프

- 신경망 전체에 걸쳐 **한 방향**으로만 데이터가 전달됨

- 신경망과 뉴런의 유사성

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/uSbS361-Syfuygn16UQskJ0bkYZFhJvOc6WJUzdK7FFrfxzQkfW_6jeT__zSxuTyGytCIUopVImkVMXxsKHuwTej2UjByAbT5RnyZfOSJNrpJz9zK13TWcDwMx5VWAD6.png" width="530"></p>

    - 노드 $\rightarrow$ 뉴런 (수상돌기)

    - 간선 $\rightarrow$ 축삭돌기

- 계층 (Layer) : 같은 선상에 있는 노드들
    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/3MwoHtiaUA77hzjxcHAk0HaejrOMFKWke0bdvUXzX9elp5a-WgbrL0V5jjgryeeXUyhqVgCUeZJc-eVVW1dpvCm5RFMTze8IS4eE1K7yrlDRMV0Uoo2itaB4Xs276OMv.png" width="450"></p>

    - 입력층(Input Layer) : 신경망에 데이터가 처음 입력되는 부분

    - 출력층(Output Layer) : 신경망을 통과한 결과가 출력되는 부분

    - 은닉층(Hidden Layer) : 입력층으로부터 입력받은 데이터에 대해 처리를 진행해, 처리 결과를 출력층으로 전달하는 부분

- **심층 신경망** (Deep Neural Network, DNN) : 은닉층의 개수가 2개 이상인 신경망

    - **딥러닝 (Deep Learning, DL) : 심층 신경망에 대한 머신러닝**


<br>

### 4.2.2. 각 뉴런의 역할

- 뉴런(노드)이 가지고 있는 동일한 규칙

1) 각 뉴런에 입력되는 값은 여러 개 가능, 그러나 출력되는 값은 오직 하나

2) 각 뉴런에 입력되는 값에는 가중치가 곱해짐

3) 출력되는 값은 활성화 함수를 통과함

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/TVFx2HmhvLS57BOK8f_P-Yxe0eQ2Yu6p1d86h0JmdSXiMc3xnlhvcgBJXePRyQW8fv-OwxslxZDmD29JKJkhqNof-oNfbjjShodaoYA7GQdy-V5_GvqGs-g0HLa_eIej.png" width="350"></p>

<br>

### 4.2.3. 가중치 곱

- 용어 정의

    - 밀집층 (Dense Layer) : 모든 입력 데이터에 대해 가중치를 곱하여 더해주는 Layer

    - 완전연결 계층 (Fully-connected Layer) : 각 층의 노드들끼리 완전하게 연결된 신경망의 Layer

- 입력 데이터 $\textbf x = [x_1, x_2, x_3]$, 각각에 대응되는 가중치 $\textbf w_i = [w_{i_1}, w_{i_2}, w_{i_3}]$ 일 때 가중치 곱은?
    
    $$\textbf x  \cdot \textbf w_i = x_1 w_{i_1} + x_2 w_{i_2}+x_3 w_{i_3}$$

- 데이터가 2차원 이상일 경우?

    - 벡터화 (Vectorization, flatten) : $n$차원 $\rightarrow$ 1차원으로 차원을 변환하는 과정

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/YsgZz3Vbi7VHZSIDn681S9Fsmv5G2O1bjAaf0fNeyYXHaNGtD4K5CXn8Rprr3LyAEce_WqxftJXWFL8KjZxx16EkVDPNPgiKZX6Rn3dI6HRbxZPOKJ1Dkp0b-HKb3MmT.png" width="250"></p>


    - 벡터화 과정의 문제점
        - 여러 행의 데이터를 하나의 행으로 이어서 합쳐주기 때문에, **데이터가 가진 공간적인 정보가 무시**됨

        - 따라서, 단순히 밀집층에서 가중치를 곱하는 방법 외에도 다른 가중치 곱이 존재 (ex. CNN (*Convolution Neural Network*))


<br>

### 4.2.4. 활성화 함수

- 뉴런 (또는 Node)에서 최종적인 값을 내보내기 전에 통과 시켜주는 함수

    - 주로 비선형 함수를 지칭

    - 선형 함수의 예시로는 항등함수가 존재

- Universal Approximation 에서 활성화 함수 : $\varphi(\cdot)$ 에 대응됨

---

**a. 활성화 함수의 종류**

> 4.2.4.a. 절에 사용된 그림의 출처는 유원준 $\cdot$ 안상준 님의 저서인 [Pytorch 로 시작하는 딥 러닝 입문](https://wikidocs.net/60683)임을 밝힙니다.

**1. 항등함수**

$$\varphi(x) = x$$
- 주로 **출력층에서 사용**됨, 별도의 계산과정이 없음

- ML 실습 4 역시 활성화 함수로 항등함수를 활용하였음!

- **회귀 문제**에서 많이 사용됨


<br> 

비록 활성화 함수의 첫 번째 예시로 항등함수를 소개하였지만, 일반적으로 활성화 함수는 **비선형 함수** 여야 한다. 선형 함수를 활성화 함수로 사용할 경우, 은닉층의 가중치 값이 바뀐 것과 같은 효과를 가져오기 때문이다. 즉, 비선형성을 부여하지 못하고 결과적으로 활성화 함수를 추가하는 의미가 사라지게 된다. 이제부터는 비선형 함수들에 대해 살펴보도록 하자.

<br>

**2. Sigmoid 함수**

$$\varphi(x) = \frac{1}{1+e^{-x}}$$

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/6dBGecdvhTh9O5738Zh2Zq88bH25cXQSD4eWRXkQIxf94VRkCBxHdF2___ltsgVQSmMLsVJ3IKc7NAnEvN_ALTvB7T-CCR1zVQxi1ZAEl3ZE3OengGPnaa8rjqA-7rO4.png" width="320"></p>

- **분류 문제**에 많이 사용됨. 이유는?

    - **출력값이 0과 1 사이의 값**을 가지므로, Binary Classification 에 활용 가능 (1에 가까울 경우 class A로 분류, 0에 가까울 경우 class B로 분류)


- 딥러닝에서는 잘 활용되지 않음 : **Gradient Vanishing** 문제를 가짐

- Gradient Vanishing

    - $x$가 0 근방의 값이 아닌, 매우 큰 값이나 매우 작은 값을 가지는 경우 **sigmoid 함수의 기울기가 0에 수렴**하게 됨.

         
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/hQyLTKcYj3hWyBFjus-1kgUMVOXe64DJmi9k2U00i2I99dJBIJrsYcXPiQVFyvqcd5Lrw4BfCrNBSxg1fFpAAm_RrX8r7ea0fk2mKAHDKFadlIfaZVGuyCtXexprahH5.png" width="320"></p>


    - 이 때 오차 역전파 과정에서 미분이 일어나고, 활성화 함수의 기울기(미분계수)가 곱해지므로 **0에 가까운 값들이 계속해서 곱해지게 됨**.

    - 결과적으로 맨 처음 층까지 Loss function의 기울기 값이 전달되지 않고, **가중치 업데이트가 일어나지 않음** (또는 느리게 일어남).

       
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/UlHv1Wi8lU1b_5coYE6uRdai3oW0nlncCmMRXB3HEF_lbiEVEk9947fP6LNgAvEB44bAHILkf7IHgYVkGfEBAuS_lXc4sh0ZoILEEkNgMykenU2IKb6KidMh1ycZCyIP.png"></p> 



- 또한 함수의 중앙값이 0이 아니라는 문제점 (중앙값은 0.5) 역시 가짐 

<br>

**3. tanh 함수**

$$\varphi(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/5gDrns75GYgr-KbQeozJ9e7rj9wFXYsEZiMv1h6TbLW6AVndkwYsxo-nnjIp3bMecDufFpLj7dQuq37N6Rl9KJ6DrkvK4hTlfWaJxFT3LiRyPROGdqemCdHLO97JnHc2.png" width="340"></p>

- Sigmoid 함수의 중앙값이 0이 아닌 문제를 해결

    - Sigmoid 에 비해 함수값의 변화가 더 큼

- 그러나, 여전히 Gradient Vanishing 문제를 겪음

    - 중앙값이 0이므로, Sigmoid 함수에 비해서는 Gradient Vanishing 이 적은 빈도로 발생

    - 결과적으로 Sigmoid 보다는 더 많이 사용됨

        - 물론, 현대의 인공 신경망에서는 두 활성화 함수 모두 사용 빈도가 낮음.
        
<br>

**4. ReLU 함수**

$$\varphi(x) = \max(x, 0)$$

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/8qGFFajhz8P8HXm61bMMh4kIkl7zTmahLXMUuRQCKqk0ItpYiQh7GqfCUX5Eggegu_EEpjIR50N2KjXHk7ZCMqzZgejQaWF8YLYu7MM6xxSyYbXi46WOO7qfM_M-bEpK.png" width="320"></p>


- 현대 딥러닝 모델에서 가장 많이 사용되는 대표적인 활성화 함수

- Sigmoid 함수의 기울기 소실 해결 가능

- 그러나 입력값이 음수일 경우 기울기가 0 ⇒ 여전히 문제 발생 : **"Dying ReLU"** 
    - Dying ReLU 해결 방안 : **Leaky ReLU** 

<br>

**5. Leaky ReLU 함수**

$$\varphi(x) =\text{max}(ax, x)$$ 

$$(0 < a < 1)$$

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/2aRyMFceyEmlMS39HUFH1Doy87NyboOAz3tMUrBQx6UwaD8ulniDzAi0crG8qfL-zjLwzjwE5Wu_P7-jgQvEpR7AtqrhfxoGvjMWlWPNVHUfy-U5Jbzpwf8vqgOvxi25.png" width="350"></p> 

- 기울기가 0이 되지 않으므로, Dyning ReLU 문제점을 해결 가능

- 그러나 새로운 Hyperparameter (음의 기울기 $a$) 를 지정해 주어야 한다는 문제점을 가짐.

<br>

**6. Softmax 함수**

- 입력값이 $x_1, x_2, \cdots, x_n$, 출력값이 $y_1, y_2, \cdots , y_n$일 때

$$y_k = \text{softmax}(x_k)=\frac{e^{x_k}}{\sum\limits_{i=1}^{n} e^{x_i}}$$


- $0 < \varphi(x_k) < 1$, $\sum\limits_{i=1}^n \varphi(x_k) = 1$ ⇒ **분류 문제**에서 **각 class 에 속할 확률**로 해석 가능

- **다중 클래스 분류 문제** 해결에 많이 사용됨

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/AgQFwoBTAf8OOoI_HGcOHZnutqn9g2xCeNwf1yn0g2dIb-4Ef--CPneOMe7abrFnCOO1T5MgDRgqWtVPdlJcZnTUf5xDEr9iJl33idnhAL9CujbAxwheVU-d6EMqu3N0.png" width="550"></p> 

---


**b. 활성화 함수의 필요성**


1) 활성화 함수를 곱해주지 않는다면, **선형 모델밖에 표현하지 못함.** 

    - 가중치를 곱해주는 과정은 입력 데이터에 대한 선형결합 (Linear Combination) 이므로.

    $$\textbf{x} \cdot \textbf{w}_i \rightarrow (\textbf{x} \cdot \textbf{w}_i) \cdot \textbf{w}_j \rightarrow ((\textbf{x} \cdot \textbf{w}_i) \cdot \textbf{w}_j) \cdot \textbf{w}_k$$

    - 따라서 비선형 함수인 활성화 함수를 곱해, **보편적인 수학 모델을 표현**할 수 있으며 입출력 데이터 간의 규칙성을 모델링 가능.

<br>

2) 활성화 함수가 없다면, 신경망의 층 수를 증가시키는 것이 의미가 없어짐. 

    - **선형 결합을 여러 층에 걸쳐 반복해도, 결국 선형결합**이므로 같은 결과를 내는 하나의 층으로 나타낼 수 있기 때문.

    - 활성화함수가 존재하면 **신경망의 층수를 증가시키는 것이 의미 있어지며**, 신경망의 층 수에 따라 비선형성이 추가됨.

    $$\textbf{x} \cdot \textbf{w}_i \rightarrow f(\textbf{x} \cdot \textbf{w}_i) \cdot \textbf{w}_j \rightarrow f(f(\textbf{x} \cdot \textbf{w}_i) \cdot \textbf{w}_j) \cdot \textbf{w}_k$$



## 4.3. 신경망의 학습

신경망 학습 과정은 크게 4단계로 구성된다.

1) 데이터 전처리

2) 신경망 모델 구축

3) 손실함수 정의

4) 손실함수 최적화 (역전파)


<br>

### 4.3.1. 데이터 전처리

여러 가지 데이터 전처리 기법 중 특성 스케일링(Feature Scaling) 또는 정규화(Normalization)에 대해서 알아보자. ML 실습 4 에서 이를 직접 구현해 볼 것이다. 아래와 같은 예시를 살펴보자.


<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/NKBszTQLDVJ2i6031fBpEIDS0qbDkceFmP546Zrazw2FahwaDD_r74hcsum2nE-1ym3PyhISrR4yq8PThl1IYsjTXvYDLllS2hwqSQuxLovNkyuav6wXlWUJ-qQeXvwW.png" width="550"></>

- 키와 몸무게의 스케일(규모, scale)이 다름 (키 값이 몸무게 값의 약 2-3배)

- 발생할 수 있는 문제점?
    - 머신러닝 모델은 단위를 제외한 값만을 가지고 문제를 해결

    - 따라서 **키 값의 중요도가 증가**하게 됨

    - 경사하강법에서, **특정 변수 (Scale 이 작은 변수, 여기서는 몸무게)에 대한 가중치의 업데이트가 잘 일어나지 않음**
        - ML 실습 4 참고 (직접 확인)

        - 순전파 과정에서, $x_i$가 $n$배가 되면 $w_i$는 $\frac{1}{n}$배가 됨 $\rightarrow$ **기울기 값 감소, 가중치 수렴 속도 감소** 

- 해결 방안? : 모든 특성 (feature)들의 scale 을 통일 시켜주기

    - 최대-최소 정규화

    - Z-점수 정규화

<br>

**(1) 최소 - 최대 정규화**

$$x'_k = \frac{x_k-\min(x_i )}{\max(x_i)-\min(x_i)}$$

$$\therefore 0\leq x_k' \leq1$$
    
- 단점 : **Outlier**가 있는 경우, 나머지 데이터들의 값이 0 또는 1에 가까워짐
    
    ex)  
    
    $$ [0, 0, 1, 1, 1, 2, 2, 2, 3, 100] $$
    
    
    $$ \downarrow$$

    $$ [0, 0, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 1]$$

<br>

**(2) Z-점수 정규화**

$$x_k' = \frac{x_k - \mu}{\sigma}$$

- Outlier의 영향을 Max-min Normalization 보다 적게 받음
- 0을 중심으로 하는 대칭적인 분포로 변환

<br>

### 4.3.2. 신경망 모델 구성 
- 층 수

- 층의 종류

- 활성화함수 등을 정의

<br>

### 4.3.3. 손실함수 정의 및 계산

여러 손실함수 (Loss Function)의 종류에 대해 알아보자.

1. SSE (Sum of Squares of Error)

    $$\mathscr L_{SSE} = \sum\limits_{i=1}^n (\hat{y_i} - y_i)^2$$  

    Q) 모델의 출력값이 [0.2, 0.5, 0.3]이고, 실제 값이 [0, 1, 0]일 때 SSE는?  *0.38*
        


2. MSE (Mean Squared Error) 

    $$\mathscr L_{MSE} = \frac{1}{n} \sum\limits_{i=1}^n (\hat{y_i} - y_i)^2 = \frac{\mathscr L_{SSE}}{n}$$ 


3. $\text{|Residuals|}$

    $$\mathscr L_{abs} = \sum\limits_{i=1}^n |\hat{y_i} - y_i|$$ 


4. CEE (Cross-Entropy Error) 

    $$\mathscr L_{CEE} = - \sum\limits_{i=1}^n y_i \log {\hat{y_i}}$$ 


### 4.3.4. 손실함수 최소화 (Training)

- Training Data 이용

- Optimizer 활용 (SGD, Adam, RMSProp 등)

    - GD 의 여러 가지 변형 : SGD, Cyclic SGD, Shuffled Cyclic SGD, etc...
    
- 오차역전파법 (Error Backpropagation) : **가중치 업데이트** 

    - 순전파 : 입력값에 가중치를 곱하고, 활성화 함수를 통과시키는 과정

    - **역전파** : 손실함수의 편미분 계수들을 이용해 가중치를 업데이트 하는 과정

        - 수치 미분 없이, 편미분 계수를 효율적으로 구할 수 있음
        
        - 자동미분 라이브러리 이용 (ML 실습 3 참고)

        - Chain Rule 기반 

        - Recap : GD at $k$ th iteration

        $$\textbf w_{k+1}=\textbf w_{k} − \delta \cdot \left.\nabla \mathscr{L} (\textbf{w}) \right|_{\textbf{w}=\textbf w_k}$$

        즉, 특정 가중치를 Update 하기 위해서는 **해당 가중치에 대한 편미분계수**를 구해야 한다.

    - Backpropagation Example 1

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/QmTrkAweLWWLcI26sj02yznU4H-LipjygVwiZiNbhfAAXkqs_NkktfPNRQ_SzK_wRfi14pAHB_HheS-CJ8KvidAljVDP92YOV8qJP6UQZwlAI11ORPopIfrObvXSjaiw.png" width="400"></p>

        위 그림에서, 가중치 $w_{z1_2}$ 와 $w_{z2_2}$ 에 대해 역전파가 어떻게 일어나는지 알아보자. 구해야 하는 값은

        $$ \frac{\partial \mathscr{L}}{\partial w_{z1_2}}, \frac{\partial \mathscr{L}}{\partial w_{z2_2}}$$

        이다. 수식을 통해 하나씩 전개해 보자. 출력층에 가까운 Layer 로부터 손실함수에 대한 편미분 계수를 구해 간다. 출력층에 가까운 Node 에서 구한 편미분계수가, 다음 층의 편미분계수를 구할 때 이용된다.

        ---


        [1] $\frac{\partial \mathscr{L} }{\partial w_{z1_2}}$ 

        
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/rkghcLoA6E-8RUlLqBUPOzlqEzeMBwumrBoCTlV8WmsP0TAk82d_Lx5gZu5n1sF8ePf-NXJA3Uc2t63NGU_HjCSysAW1DPPaI2BvNBW9cv2MBWsN2qGkPdeVMVF3Wurd.png" width="630"></p>


        $$\frac{\partial \mathscr{L} }{\partial w_{z1_{2}}} = \frac{\partial \mathscr{L} }{\partial z_1} \frac{\partial z_1}{\partial w_{z1{2}}}$$


        이 때, $z_1 = \varphi(y_1) w_{z1_{1}}+ \varphi(y_2) w_{z1_{2}}+ \varphi(y_3) w_{z1_{3}}$ 이므로

        $$\frac{\partial z_1}{\partial w_{z1_{2}}} = \varphi(y_2)$$


        따라서, $$\frac{\partial \mathscr{L} }{\partial w_{z1_{2}}} =\frac{\partial \mathscr{L} }{\partial z_1} \cdot \varphi(y_2)$$

        ---

        [2] $\frac{\partial \mathscr{L} }{\partial w_{z2_2}}$


        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/LWWnPr3AUBk_65UrWWr2xcwGO-Koxv-J5f6ZE4uBUggzFJ51lEmvHN6Qp72dryKtTwL6ViiZW8MAolNCBnYVcrKhbST5iv7UCgXosxoryLjzTk60mNoQSPAo_WQ62JGC.png" width="630"></p>

        $$\frac{\partial \mathscr{L} }{\partial w_{z2_{2}}} = \frac{\partial \mathscr{L} }{\partial z_2} \frac{\partial z_2}{\partial w_{z2_{2}}}$$

        이 때, $z_2 = \varphi(y_1) w_{z2_{1}}+ \varphi(y_2) w_{z2_{2}}+ \varphi(y_3) w_{z2_{3}}$ 이므로

        $$\frac{\partial z_2}{\partial w_{z2_{2}}} = \varphi(y_2)$$

        따라서, $$\frac{\partial \mathscr{L} }{\partial w_{z2_2}} =\frac{\partial \mathscr{L} }{\partial z_2} \cdot \varphi(y_2)$$

        [1], [2] 에서 최종적인 값을 구하기 위해서는

        $$\frac{\partial \mathscr{L} }{\partial z_1}, \frac{\partial \mathscr{L} }{\partial z_2}$$
        
        의 값을 구해야 한다. 역시 Chain Rule 을 이용해 구하면 아래와 같다.


        ---

        [3] $\frac{\partial \mathscr{L} }{\partial z_1}, \frac{\partial \mathscr{L} }{\partial z_2}$

        $$\frac{\partial \mathscr{L} }{\partial z_1} = \frac{\partial \mathscr{L} }{\partial \varphi(z_1)} \cdot \frac{\partial \varphi(z_1) }{\partial z_1}$$

        $$\frac{\partial \mathscr{L} }{\partial z_2} = \frac{\partial \mathscr{L} }{\partial \varphi(z_2)} \cdot \frac{\partial \varphi(z_2) }{\partial z_2}$$

        ---

        이와 같이 가중치 $w_{z1_2}$ 와 $w_{z2_2}$ 에 대한 편미분계수를 구해 보았다. 이제는, 조금 더 복잡하게 가중치 $w_{y2_2}$ 에 대한 편미분계수를 구해 보자.

        [4] $\frac{\partial \mathscr{L} }{\partial w_{y2_2}}$ 

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/l6okzYDKcFkwY8OJ4sjg4YgkGSwiy69MK6QsHdnJDAAA4L3c2diRani2i2L5vGgqCjTb0mhAvxtlIZwWn5jhuvN_GPhapVJ9HuZ9Kk7u3K38rpBUo8TxliTUFd5PZDY-.png" width="630"></p>

        우선, Chain Rule 에 의해

        $$\frac{\partial \mathscr{L} }{\partial w_{y2_{2}}} = \frac{\partial \mathscr{L} }{\partial y_2} \frac{\partial y_2}{\partial w_{y2_{2}}}$$

        가 성립한다. 이 때

        $$\frac{\partial y_2}{\partial w_{y2_{2}}} = x_2$$
 
        임을 이용하자. 이제 

        $$\frac{\partial \mathscr{L} }{\partial y_2}$$

        를 구해보자. Chain Rule 에 의해

        $$\frac{\partial \mathscr{L} }{\partial y_2} = \frac{\partial \mathscr{L} }{\partial \varphi(y_2)} \cdot \frac{\partial \varphi(y_2) }{\partial y_2}$$

        이 성립한다. 즉, 

        $$\frac{\partial \mathscr{L}}{\partial \phi(y_2)}$$ 

        의 값을 구해야 한다.

        ---


        [6] $\frac{\partial \mathscr{L}}{\partial \varphi(y_2)}$

        By chain rule,

        $$\frac{\partial \mathscr{L}}{\partial \varphi(y_2)} = \frac{\partial \mathscr{L}}{\partial z_1} \frac{\partial z_1}{\partial \varphi(y_2)} + \frac{\partial \mathscr{L}}{\partial z_2}\frac{\partial z_2}{\partial \varphi(y_2)}$$

        In the same way,

        $$\frac{\partial z_1}{\partial \varphi(y_2)} = w_{z1_{2}}, \; \frac{\partial z_2}{\partial \varphi(y_2)} = w_{z2_{2}}$$

    
        Therefore, 
        
        $$\frac{\partial \mathscr{L}}{\partial \varphi(y_2)} = \frac{\partial \mathscr{L}}{\partial z_1} \cdot w_{z1_{2}} + \frac{\partial \mathscr{L}}{\partial z_2} \cdot w_{z2_{2}}$$ 


        이 때 

        $$ \frac{\partial \mathscr{L}}{\partial z_1}, \frac{\partial \mathscr{L}}{\partial z_2}$$ 

        의 값은 [3] 과정에서 구하였으므로, 해당 값을 이용해 최종적인 편미분계수를 구할 수 있다.


        ---

    - [Backpropagation Example 2](https://github.com/frogyunmax/OUTTA_2022AIBootcamp/blob/main/ML_Chap2-4_NN_Lecture.pdf) : pp. 65 - 80
 
<br>

- *Remark 1 :* 위 과정을 'Local Gradient' 와 'Downstream Gradient' 를 이용해 더 간결하게 일반화 할 수 있다. 관심이 있다면 구글링을 통해 조사해 보자.

- *Remark 2 :* Use Matrices 
    $$\textbf{z} = [z_1, z_2]= [\varphi(y_1)w_{z_{11}} + \varphi(y_2) w_{z_{12}} + \varphi(y_3) w_{z_{13}}, \varphi(y_1)w_{z_{21}} + \varphi(y_2) w_{z_{22}} + \varphi(y_3) w_{z_{23}}]$$  


    순전파 결과를 행렬을 이용해 표현하면
        
    $$Z=(z_1 \; z_2) = YW$$

    where
    
    $$(Y = \begin{pmatrix}\varphi(y_1) & \varphi(y_2) & \varphi(y_3) \end{pmatrix}, W =\begin{pmatrix} \textbf{w}_{\textbf{z1}} & \textbf{w}_{\textbf{z2}} \end{pmatrix} = \begin{pmatrix} w_{z_{11}} & w_{z_{21}} \\\ w_{z_{12}} &w_{z_{22}} \\\ w_{z_{13}} & w_{z_{23}}\end{pmatrix})$$
                    
    이 때, 행렬곱을 이용해 밀집층의 역전파를 계산하면 아래와 같다.

    $$\frac{\partial \mathscr{L}}{\partial Y} = \begin{pmatrix} 
    \frac{\partial \mathscr{L}}{\partial \varphi(y_1)} & 
    \frac{\partial \mathscr{L}}{\partial \varphi(y_2)} &
    \frac{\partial \mathscr{L}}{\partial \varphi(y_3)} 
    \end{pmatrix}
    = \begin{pmatrix} 
    \frac{\partial \mathscr{L}}{\partial z_1} & 
    \frac{\partial \mathscr{L}}{\partial z_2}
    \end{pmatrix}
    \begin{pmatrix} 
    w_{z1_1} & w_{z1_2} & w_{z1_3} \\\
    w_{z2_1} & w_{z2_2} & w_{z2_3}
    \end{pmatrix}
    = \frac{\partial \mathscr{L}}{\partial Z}W^T$$

    $$\frac{\partial \mathscr{L}}{\partial W} = \begin{pmatrix} 
    \frac{\partial \mathscr{L}}{\partial w_{z1_1}} & 
    \frac{\partial \mathscr{L}}{\partial w_{z2_1}} \\\
    \frac{\partial \mathscr{L}}{\partial w_{z1_2}} & 
    \frac{\partial \mathscr{L}}{\partial w_{z2_2}} \\\
    \frac{\partial \mathscr{L}}{\partial w_{z1_3}} & 
    \frac{\partial \mathscr{L}}{\partial w_{z2_3}} 
    \end{pmatrix} = \begin{pmatrix} \varphi(y_1) \\\ \varphi(y_2) \\\ \varphi(y_3) \end{pmatrix}
    \begin{pmatrix} 
    \frac{\partial \mathscr{L}}{\partial z_1} & 
    \frac{\partial \mathscr{L}}{\partial z_2}
    \end{pmatrix}
    = Y^T \frac{\partial \mathscr{L}}{\partial Z}$$

<br>



## 4.4. 순전파와 역전파의 반복

- **1 Iteration** 의 정의

    - 순전파 + 역전파 

    - 손실함수에 대한 역전파, Optimizer 통해 가중치 업데이트

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/L6VmbEbMz94FiQbTU6kfkP-_ZEmJ7Rqjn3o8fiq27AkRx-BBk82xV6bWBiF3z_VcxJpSMQgMWrZte7jYSURaz3SwlEX-Vjeh4IO2zZ-gKSVIxxdGMmvi_JuVjuWZ_QbH.png" width="220"></p>

- **batch** 의 정의 : 신경망에 한 번에 입력해주는 데이터 묶음

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/i4tmg6xERBEB0LD8mFU6RV61Eq-vU8omLNo6E70rccQHb7W6MBG-BNylthrAKKSopWPKaYxaYHsyPiypaaGhlz6ezLOhfIZVj2jcfNgjpy5gjXjTXGs3IsDuMGY6w5_Q.png" width="450"></p> 

    - batch size : 1개의 batch에 들어 있는 데이터의 개수 $(M)$

    - batch 구성 방식 : 전체 데이터에서 $M$개 랜덤 추출


- **1 epoch** 의 정의 

    - 전체 데이터가 $D$개, batch size가 $M$일 때 $D/M$ 번의 iteration을 통해 가중치 업데이트가 1회 일어나는 과정

    - 1 epoch = $D/M$ iterations

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/QKWLwy_ZWIZpdLOxLXyixqWZu0OYEtnUhpAF28ZN4BeL6q95UoM1zGGkThLD_Qw4DTw6uYNtdj0S0eKJPfqMiBHItYukreGuUgBdNhH_1_jQlSN8NTWSygGcZFwC6Obj.png" width="400"></p> 


- *Remark :* batch 를 사용할 때의 이점

    1) 최적의 가중치를 향해 연속적이고 부드럽게 이동할 수 있음

        손실함수의 그래디언트는 **데이터 별로 상당히 큰 차이**가 있을 수 있다. 따라서 개별 데이터에 대해 가중치 업데이트를 각각 진행할 경우에는 **가중치가 역동적으로 움직이는 문제**가 발생할 수 있다. 그러나 batch size가 M인 batch의 평균적인 손실 함수에 대해 생각 하는 경우, 가중치 업데이트의 **역동성이 완화되어 좀 더 안정적이게 된다**.


    2) 메모리의 효율적 사용, 학습 속도 향상

        데이터 전체를 한 번에 신경망에 입력해주는 경우, 데이터셋을 이루는 **모든 데이터에 대한 오차**를 메모리에 전부 저장해야 한다. 이로 인해 메모리 공간을 많이 사용하여 **학습 속도가 저하**된다. 그러나 batch로 **분할하여 학습하면 메모리 공간 절약**에 의해 학습 속도가 향상된다.

<br>

## 4.5. 하이퍼파라미터 (Hyperparameter)
- 딥러닝 모델의 성능에 영향을 주는 변수

- 사용자가 직접 설정, 입력

- Weight initialization

- Learning rate

- Early Stopping

- Depth of Layer 

- Numbers of nodes in one Layer

- Activation Function

- Loss Function

- Optimizer

- batch size

- patch size

- Dropout ratio

- etc...

## References
>별도의 출처가 표기되지 않은 Figure의 경우 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》에서 가져왔거나 원본을 기반으로 재구성 된 Figure 입니다.
