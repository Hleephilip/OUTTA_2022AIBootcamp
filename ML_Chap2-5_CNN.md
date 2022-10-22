# [ML/DL] 2장 - (5). Convolutional Neural Networks

이 자료는 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》 를 바탕으로 제작되었습니다. 

Remnote 자료는 [링크](https://www.remnote.com/a/-ml-dl-2-5-cnn-/6353acb8dbe54f4b4944f391)를 통해 확인하실 수 있습니다.

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)


## 5.1. 컴퓨터가 이미지를 인식하는 방식

**5.1.1. 비트맵 이미지**

- 각 위치의 픽셀들이 어떤 색상 정보를 담고 있는지 저장

- 흑백 이미지의 경우?

    - 검은색 0, 흰색 255를 기준으로 정도를 표현

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/Vg75RFu7stf0ql-o4FCZJ5MfelPRuJd4nxdIdK69uwtNp3nJyK0KguZZLqKA2Kn27T-XUAQJliq_sHkTUozlxtT5XhESSN9TLeeBwfsl-UofLWkd5wfbja1XRi-Bm58s.png" width="200"></p>

- 이미지가 흑백이 아닌 경우?

    - **빛의 3원색** (R, G, B) 이용

    - 이미지가 흑백이 아닌 경우, 색을 만들어내는 데 빛의 삼원색(빨강, 초록, 파랑)을 얼마나 합성시켜야 하는지를 나타내면 된다. 각각의 색이 합성되는 정도를 표현할 때, 합성되지 않음을 0으로, 최대한으로 합성됨을 255로 나타낸다.

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/eCc7LMAYSmAxjF_pmq1A-3RomjdwCWPvvbt5YE6DJ8oYmhb5aTbbhCk7SNmKN4VAgBaanI7E5BZsvwl2oGBj8CY87zVuBRWdKxGM0h6gGV7myJV9fdNiq1V6nksNKYkM.png" width="220"></p>

- 채널

    - 이미지를 구성하는 요소, **색상과 같은 의미** (R/G/B 채널)

    - 이미지 픽셀을 입력으로 받아, 출력으로 픽셀을 내보내는 사상 (Mapping)

    - 이미지로부터 **추출할 수 있는 어떠한 정보를 격자 형태로 배열해둔 것** 

**5.1.2. 벡터 이미지**

- 점들을 연결하여 선을 만들고, 선들을 연결하여 면을 형성되는 정보 자체를 저장하는 방식

<br>

## 5.2. 밀집 신경망에서 CNN으로

- 이미지 처리 분야에 밀집 신경망 (FC Layer)이 사용 불가능한 이유?

    - 밀집 신경망의 경우 1차원으로 입력을 받음
        
        - 2차원 데이터의 경우 Flatten 된 형태로 입력됨

    - 즉, **이미지의 공간적 정보를 나타내지 못함!**

    - CNN : 공간적 정보를 유지하며 학습하는 신경망



## 5.3. CNN의 구조

<p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/nQ22zwcLWam4esOHc3e9tUlvdSrby9whVbMD9LcY1ttj3vGtW0D2he9PKcaXX9falXpqfhNScKHVApbMiIXkx2nCRbIOk4Q2gFchS0ClHosbaqrdxyqzfOUFSvhdVlwn.png" width="680"></p>


## 5.4. 합성곱 신경망
- 필터 (Filter) = Kernel

    - "합성곱 연산을 수행시킬 필터"

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/kuOa08j2JPDRLjmnhsoGvNqfbi6DtLgd3jA74XyQhILCyG9RLvB5fhRFFXqTC7jyjt1DVW3WWmQod2IFU9FgjRiibwK62E0qxfhD5npoJ2gYXE60klxcId88kYlDL9pJ.png" width="90"></p>

    - Filter Size : Hyperparameter

- Convolution 연산

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/TI9miH9fSnTx0wrsUduMpFirEx_hFxt5GmzFN3FJP1d6chfQlPMVKuxUbSYkacvaVeT8olZ9LPwqb1NTl-uYWIM061-ttmCQSvzeF4-qgmLlNACtXAOEtrJgW-AnLzHv.png" width="480"></p>


    - 편향 (bias)이 존재하는 경우

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/HnKrc5rwLCqoHFaOVC2JC7I-6s-QGSXh8QXVCVnKBE-f6G1Pix8rlTkb55NWf03iNTd2Vu-ar7plteyAHXDgLLjg-V0vzS_g2a8c-1M2B3s8oJTqMOJrlc5feTG-TxYb.png" width="750"></p>


- Stride
    - Filter 가 이동하는 칸 수 (필터 연산을 적용할 위치의 간격)

    - Stride = $1$ 의 경우

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/nc9_cqlSIxiQaZvYxNLkNbzHauHwJ3QS_CfUwrQuYzbToCBSqOvjT3jQOswTVJQx4efxCIWZyCzHQVNcdhZpOvQ4YjBma0EKJ6CH8kDx8qKgwJOlUR4HIoYpqJF8HxQF.png" width="470"></p>  


    - Stride = $2$ 의 경우

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/OOsH_MTegCgoC8EngUGDpCAxGAUy_tJknQXYa-AYJc1Wopi7diuPLFsSsR3h5bcTAMKoFKQ7TUeRb-evFi8TiDs60wW6pNocC6l3jxDLg53nPwvrXSKreKplRVyGc7tF.png" width="430"></p>
    
- Padding

    - 가장자리쪽의 픽셀은 이미지 안쪽의 픽셀보다 **Filter가 거쳐 가는 횟수가 적음** 

    - 따라서 가장자리 데이터는 합성곱 신경망의** 순전파를 진행할 때 약하게 전달**될 것이며, 가장자리에 중요한 데이터가 포함되는 경우 **학습 효율에 악영향**을 미침

    - **패딩** : 합성곱 연산을 거치기 전, **입력 데이터 주변을 특정 값 (0)으로 적절히 채워 넣는 기법** 
        
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/g2gsh0xfO5WTjA_Diw74syrwKnVLiih7hNsezuMzKfiO86HMVFsf3lrJoca9bb640f0iEz0ytLx0GFrjCB47QVVUVdwJF_yu8A5rJGBycOcngpcdq0oENKPPIqdIKKES.png" width="670"></p>
    
    
    - 패딩을 사용한 후, 동일한 필터 연산을 수행하면 입력 데이터의 **가장자리와 내부 픽셀은 동일한 횟수의 필터 연산**을 거치게 됨

    - 패딩의 이점 (2가지)
        1) **가장자리 정보 소실 방지**

        2) 입력 이미지의 크기를 증가시키는 효과 ⇒ **출력값의 Size가 너무 작아지는 문제 방지** 

- 출력값의 Shape 계산하기

    - 입력값의 (채널 1개 당) Shape : $(H_{in}, W_{in})$ 
    
    - Filter Shape : $(f_1, f_2)$

    - Padding : $P$

    - Stride : $S$

    - 출력값의 Shape : $(H_{out}, W_{out})$ 

        $$H_{out} = \frac{H_{in} + 2P - f_1}{S} + 1$$
    
        $$W_{out} = \frac{W_{in} + 2P - f_2}{S} + 1$$

        - cf) Stride 역시 가로/세로 방향에 대해 다르게 설정할 수 있다. 여기서는 각 방향으로 움직이는 정도가 같다고 가정하고 공식을 유도하였다.



- 2개 이상의 채널로 표현되는 이미지에 대한 합성곱 계층

    - 입력받는 데이터 채널 수와 필터 채널 수를 동일하게 설정

        - ex) RGB : 입력 데이터 채널 3개 $\rightarrow$ 필터 채널 **3개**

    - **필터 개수는 1개** / 채널 수 **3개** (두 개는 다른 개념이므로 주의할 것!)
    
        - Tensor 형태로 나타낸다면 Filter 의 Shape 는
        
            $$ 1 \times 3 \times f_1 \times f_2$$


        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/R6V9_yIQtqYx7590YtwSRWzW5anshRdKRuHg0V0lw375OV-Bkt_WZXMfsxaMtUFhwZLNzWUli1S5ETCO0F8G3P1ckqn3x2wvN_PPn5PXJDGRTyIQXy67Y1umskLmqV-u.png" width="610"></p>
    

- 여러 개의 필터를 사용하는 합성곱 계층

    - 필터 하나에 대해 합성곱 연산을 적용한다면 **채널이 1인 출력값**을 얻음

        - 입력값의 Shape 가 $(C_{in}, H, W)$ 이고 Filter 1개의 Shape 가 $(C_{in}, H, W)$ 이라면, 각 채널별로 컨볼루션 연산을 진행한 뒤 같은 위치(픽셀)의 값들을 모두 더하게 됨. $\rightarrow$ 이게 채널이 1인 출력값이 되는 것.

        - Filter 의 개수가 여러개일 경우, 이러한 작업이 Filter 의 개수만큼 일어나고 각각의 결과들이 모아지게 (Concat)됨. 아래의 필터 뱅크에서 이어서 설명.
    
    - **필터 뱅크 (Filter Bank)** : 여러 개의 필터를 모아 놓은 것

        - 입력값의 Shape 가 
        
            $$(C_{in}, H, W)$$
        
            이고 이 입력값들에 대한 필터 1개의 Shape가 $(C_{in}, f_1, f_2)$ 일 때, 필터가 $N$ 개 있다면 Filter Bank 의 Shape 는

            $$(N, C_{in}, f_1, f_2)$$

            가 된다. 이 때 출력값의 채널 수는 $N$ 이 되므로, $N=C_{out}$ 이다. 결과적으로 Filter Bank 의 Shape 는 아래와 같이 표현할 수 있다.

            $$(C_{out}, C_{in}, f_1, f_2)$$

            이 때 출력값의 Shape는

            $$(C_{out}, H_{out}, W_{out})$$ 

            이 되며 $(H_{out}, W_{out})$ 의 값은 위에서 정의한 것과 같이

            $$H_{out} = \frac{H_{in} + 2P - f_1}{S} + 1$$
    
            $$W_{out} = \frac{W_{in} + 2P - f_2}{S} + 1$$

            이다.


    - 각각의 필터를 적용한 결과를 모아주기 (Concatenation)

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/wiP6OjvSdosRw0aZPGJgLQ1_HTi6pKHdb89EqYqshSDa1h0v05EeQ3VRfNnmcrVPFDNNZwbiQh840ffAYrpeBAt2HgfJdAwXICnWVQMwJGOND-6Anwk90iKx4FeM0NUt.png" width="580"></p>


    - 편향(bias) 을 적용할 경우?

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/YLdd-TtNlP8VdP8qwYN6CbPEBA_xlfQYuzygA1OLnJN2750ROLGkRxgPt3TJEcyZT26sVqNC-hE9e2-wfUK4N37l54_qR1aalxbJdxlzpliJUDUDIHKYk5mGf_QkTxjS.png" width="740"></p>


    - 정리하자면, **입력층에 대해 사용한 필터 수 $(N_{filter}) =$ 출력층의 채널 수 $(C_{out})$** 


- 합성곱 신경망의 가중치가 Update 되는 방식 : **필터와 편향값의 가중치가 갱신됨**


## 5.5. 풀링 계층(Pooling Layer)

- 데이터의 크기 (Size)를 줄이기 위해 사용하는 층

    - 단, 채널 수는 변하지 않음.

- 별도의 **학습이 일어나지 않음** 

    - 풀링 계층의 경우 학습 가능한 (Trainable) 파라미터가 없음.

- $n \times m$ 풀링 : $n \times m$ 영역을 그 영역을 대표하는 원소 하나로 축소하여 나타내는 연산

    - Max Pooling

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/PG6QxYZh7cIpFBF9DORUNdEvRM44FhxV7A-tuCbfg6UnKvN5_sZesCiWmgimd3U-Rmvs2OpUjSwKKOyHxWzZLk-yqE_C2F4LUi-Yd4Bv-Z_aJNmm42XHgJwOiaxGs8_s.png" width="380"></p> 

    - Average Pooling

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/HwqIae6YjzGcCYN7ZaicqmWApaJeUs3Ui8eKdyXHosyEmWHkjNqJjbCAbK9ivLrd6Y0QBBuSTw8644Wp-ESTpY9qhKGcw6Dk4vKHCvZ68Y_66upovsuTxBDpEyLOSbSW.png" width="380"></p> 



- 입력 데이터의 변화에 민감하지 않음 (이유는?)

    - 최대 풀링을 수행하는 경우 최댓값이 변화하지 않는 이상 결과값은 그대로일 것이며, 평균 풀링을 수행하는 경우에도 값의 변화가 평균에 반영되며 작게 나타날 것이다.



- 여러 채널의 데이터의 경우?
    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/XmzPAmhx3aT4vVkHbS5e_C6aAV93WEdfdWncLgKlmg4Oj3_6EtAc52VMmjg3LuWq6K80a-LrWwzjxbYiQl1oEFMrWqoGOTJmbYxjipCuwxzPaci6dD5VxgTQxJej6dUW.png" width="460"></p>



## 5.6. CNN 학습시키기

- 순전파 과정 : 입력 ⇒ 합성곱 계층 ⇒ 활성화 함수 ⇒ 풀링 계측 ⇒ (반복)

- **벡터화 (Vectorization, Flatten)**

    - 이미지 **분류 작업**을 진행하기 위해, **밀집층에 입력하기 위한 형태인 일렬 데이터**로 나타내는 과정

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/xC-VTMg9-URGW1CqHUsxkY8mcD_sRGiWLYdbSjuKT9kGPN7MySZRGFWRp5EnLaMqUbI8MByrsBLfj_bd7TwL_MoXu0wIngcIQndCi6CvHcyWgTh2P5eALBWvxEZMp4Hm.png" width="280"></p>

    - Flatten 이후 사용하는 활성화 함수? : 소프트맥스 함수

- 손실함수를 통해 모델을 평가하면서, 모델 파라미터인 **필터, 그리고 밀집층의 가중치와 편향을 갱신**

- 최종적으로, 합성곱 계층에서 연산에 사용되는 필터 = **이미지의 특징을 추출**해주는 필터

- 층 깊이에 따른 추출 정보

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/PdY5stb2VIROKnzk6k-PkYCqDKlwDC4pER9YY3cVRLqg3ICL99NazEGbDcGYie0uRXzkAUmmkCSv3fmmQLSv4So9WDCip2mnCf1Vw-Yg0Cp346vUCEwtChS0symWIi5c.png" width="800"></p>

    - 합성곱 계층을 더 많이 구성할수록 **더 추상적인 정보**를 얻을 수 있음

## References
- OUTTA, 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》

>별도의 출처가 표기되지 않은 Figure의 경우 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》에서 가져왔거나 원본을 기반으로 재구성 된 Figure 입니다.
