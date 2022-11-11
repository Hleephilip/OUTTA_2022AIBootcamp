
# [NLP] 2장. 기본 언어 모델

이 자료는 인공지능 교육 비영리단체 OUTTA 에서 출판한 《인공지능 교육단체 OUTTA 와 함께 하는! 자연어 처리 첫 단추 끼우기》 및 유원준 $\cdot$ 안상준님께서 집필하신 《딥 러닝을 이용한 자연어 처리 입문》을 바탕으로 제작되었습니다. 

원본 Remnote 자료의 경우 [링크](https://www.remnote.com/a/-nlp-2-/636dfc7c43766f7796d82518)를 통해 확인하실 수 있습니다.

Made by [*Hyunsoo Lee*](https://github.com/frogyunmax) (SNU Dept. of Electrical and Computer Engineering, Leader of Mentor Team for 2022 Spring Semester)

## 2.1. 텍스트 분석과 토큰화

### 2.1.1. 텍스트 분석
- 언어의 대표적인 2가지 형식 : 말 (Speech) - 아날로그, 텍스트 (Text) - 디지털

- 분석의 기준 : 각각 글자 / 단어
- 단어의 복잡성에 따른 기준 설정


    - 품사 < 통사구조 < 개체 간 관계 < 술어
    
    <br>

    <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/7CzA50OjjyqbElV_GtWtTxH9375ScKTaEONIdfda_oPWyUyguqb9u2i7SoGhp4wc50u2GgS7uyfJ2c58FMF20fnLMpjSpO04XTnDsB-P08t7_paJAPzanVomU7MKolhw.png" width="500"><figcaption>- 아래로 갈수록 복잡성이 증가</figcaption></p>



### 2.1.2. 말뭉치 (Corpus)와 토큰화 (Tokenization)

- 말뭉치 (Corpus) 

    - 텍스트를 컴퓨터가 읽을 수 있는 형태로 모아 놓은 자료

    - 처리하고 분석할 자연어 그 자체

- 토큰화 

    - Corpus를 분석하기 위해 작은 단위 (= 토큰, token)로 나누는 과정

    - 토큰화 기준 : 형태소, 단어, 어절 등

### 2.1.3. 불용어 (Stopwords)

- 불용어 

    - 문장 내에 자주 등장하면서 **중요한 문법적 기능을 수행하**지만 실질적인 **의미를 전달해주지 못하는** 단어


    - Q. 아래 그림에서 불용어는?

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/0ks74BScT999wikldJJL4GCU27CNZt9weA4qAzKeW6yBfS6s7skmYsOqxhlSvxuP8NkSItQ-scvIxcBo0_xCT8ELq7dU_y7rqpT7XBW2NLOM2GVCWnlBhsQK0SgvszWZ.png" width="420"></p>


        - the of, is, a

    - 텍스트를 분석할 때는 **불용어를 최대한 제거**하고, **중요한 의미를 전달하는 단어만을 이용**해야 함.
    
    - 한국어 불용어 예시 (참고)
        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/plzZx7SXv_FrKJjLOynEqqfLMYOJ-NP3ICUDaCZky_LeYUUL07LvxHYSNTvMJz7hV7oF-GQc8Wc9KuFk9HqqiE6mr6_WpEzMAjeDQKtustZh9rJAnKAq4Juxen6WVEtv.png" width="450"></p>

### 2.1.4. 인코딩 (Encoding)
- (정수)인코딩

    - 텍스트의 단어에 숫자를 대응시켜 주는 작업

        - 예시 : 빈도수가 높은 단어부터 Numbering, 사전 순으로 Numbering 등..
    
    - Python 구현

        1. Use Dictionary

        2. Use Counter

        3. Use Keras

- (정수)인코딩이 필요한 이유 (2가지)

    1. 컴퓨터가 받아들일 수 있는 데이터 형식은 **텍스트가 아닌 숫자**이기 때문
    
    2. 문서를 **정량적으로 분석**한 후 **정성적인 해석** 가능

## 2.2. 조건부 확률 

### 2.2.1. 조건부 확률의 정의

- 사건 B가 **일어났을 때** 사건 A가 일어날 조건부 확률 : $P(A|B)$ 

    $$P(A|B) = \frac{P(A\cap B)}{P(B)}$$ 

### 2.2.2. 확률의 곱셈법칙

- 교집합에 해당하는 확률을 조건부확률의 곱으로 나타낸 것
   
    $$P(A \cap B) = P(A)P(B|A) = P(B)P(A|B)$$ 

    $$P(A \cap B \cap C) = P(A)P(B|A)P(C|A \cap B)$$ 



## 2.3. 통계적 언어 모델 (SLM)
- 전통적인 접근 방법 

### 2.3.1. SLM 과 조건부 확률

- 문장이 나타날 확률을 조건부 확률을 이용해 표현

    - 특정 단어 A 뒤에 특정 단어 B가 위치할 확률을 계산할 수 있으며, 이어질 단어를 예측할 수 있음

        - ex) 
        
            $$P(\text{The black cat})= P(\text{1st word is The}) \times P(\text{black place after The}) \times P(\text{cat place after The, black})$$ 


    - $P(\text{The black cat})$ 을 조건부 확률을 이용해 나타내면?

        - let $\text{The} = w_1, \ \text{black} = w_2, \ \text{cat}=  w_3$
            
            $$P(w_1 \cap w_2 \cap  w_3) =P(w_1) \cdot P(w_2 | w_1) \cdot P(w_3 | w_1 \cap w_2 )$$ 
    - 일반화
        
        $$P(w_1, \ w_2, \ \cdots, w_n) = \prod_{i=1}^{n} P(w_n|w_1, \ w_2, \ \cdots, \ w_{n-1})$$ 


- Count 기반 접근

    - '해당 Sentence 가 나타난 횟수' 에 기반해서 조건부 확률 계산
    
        $$P(w_n|w_1, \ w_2, \ \cdots, \ w_{n-1}) = \frac{N(w_1, \ w_2, \ \cdots, w_{n-1}, \ w_n )}{N(w_1, \ w_2, \ \cdots, w_{n-1} )}$$ 

    - Count 기반 접근의 한계점 : Sparsity Problem

### 2.3.2. Sparsity Problem

- **Sparsity Problem**: 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못하는 문제  


    - 학습 데이터셋이 매우 크지 않은 경우 문장에 대한 확률이 0이 되며, 현실에서의 확률을 잘 표현하지 못하게 됨

    - Example)
        - My cat has brown eyes 라는 문장에 대해, 모델을 학습시키기 위한 데이터 (Corpus)에 "My cat has brown eyes" 가 없다면 해당 문장이 발생될 확률은 0으로 계산됨

        - 그러나, "My cat has brown eyes" 라는 문장은 현실 세계에서는 존재

- 해결 방안

    - Size가 매우 큰 Dataset 사용

    - N-gram 모델 사용 ⇒ 그러나 근본적인 해결책은 아님

    - 따라서, NLP 분야에서 언어 모델의 focus 는 통계적 언어 모델이 아닌, 인공 신경망을 이용한 언어 모델로 이동

        - **Perplexity** 비교를 통해, 모델 간 정량적인 성능 비교 가능

        - Facebook AI 연구팀에서는 인공 신경망 언어 모델과 N-gram 모델의 Perplexity 를 비교

            - RNN, LSTM, Ours (인공 신경망을 이용) vs. N-gram models

            - perplexity 가 더 낮은, 인공 신경망 모델들 (RNN, LSTM, Ours)의 성능이 5-gram 모델에 비해서 더 높음

## 2.4. N-gram 모델

### 2.4.0. Motivation

- Sparsity Problem 을 해결할 수 있는 방법으로는 어떠한 것이 있을까?

    - 카운트 시, 참고하는 단어의 개수를 줄이는 방법

        - Main Idea : 
        
            $$P(\text{is} | \text{An adorable little boy}) \simeq P(\text{is}|\text{little boy})$$ 

    - 앞 문장의 전체 단어 대신 임의 개수의 단어만을 포함하는지 check ⇒ 가지고 있는 데이터에서, 해당 문장을 발견할 확률이 높아짐

### 2.4.1. n-gram 개념

- n-gram : 주어진 말뭉치 (Corpus)에서 **연속된 n글자/단어의 집합** 

    - $n$ : 하나의 토큰에 포함시킬 글자/단어 수

    - $n = 1$ : Unigram

    - $n = 2$ : Bigram

    - $n = 3$ : Trigram

- n-gram을 이용한 토큰화 Example

    - Example 1 : "The black cat eats a muffin."

        - Unigram $\rightarrow$ The / black / cat / eats / a / muffin

        - Bigram $\rightarrow$ The black / black cat / cat eats / eats a / a muffin

        - Trigram $\rightarrow$ The black cat / black cat eats / cat eats a / eats a muffin

    - Example 2 : "An adorable little boy is spreading smiles"

        - unigram $\rightarrow$ an, adorable, little, boy, is, spreading, smiles 
        
        - bigram $\rightarrow$ an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
        
        - trigram $\rightarrow$ an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles

        - 4-grams $\rightarrow$ an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

### 2.4.2. n-gram 모델에서 확률 계산
- 현재 단어 앞의 $n-1$개의 단어만 예측에 사용되고, 그 이전의 단어들은 모두 무시된다.

    $$P(w_i|w_1, \ w_2, \ \cdots \ , w_{i-1})=P(w_i|w_{i-(n-1)}, \ \cdots \ , w_{i-1})$$

- ex) The black cat eats a _______
    
    - 4-gram model 의 경우, "cat eats a" 의 단어만 이용하며 그 앞의 "The black" 은 고려되지 않음.

        $$P(w|\text{cat eats a})= \frac{N(\text{cat eats a }w)}{N\text{(cat eats a)}}$$ 

- $n - 1 =$ n-gram 모델에서 **특정 단어를 예측하는 데 살펴볼 앞 단어의 개수**

    - 마르코프 가정 : 특정 글자를 볼 확률은 **그 글자 앞에 존재하는 한정된 개수의 글자에 의해 결정된다.** 

### 2.4.3. n-gram 모델의 한계점

- 희소 문제
    - 앞에 존재하는 모든 단어를 count 할 때 보다는 빈도가 적지만, 여전히 희소 문제가 발생할 수 있음

- 모델 성능 하락

    - 전체 문장을 고려하지 않고 인접한 $n-1$ 개의 단어만을 고려함 ⇒ 정확도가 낮아짐

    - 전혀 다른 의미를 가지는 문장이, n-gram 모델에 의해서는 유사한 의미를 가지는 문장으로 해석될 수도 있음.

- $n$값 선택에 대한 Trade-off

    - Large value
    
        - pros : 높은 정확도

        - cons : 희소 문제의 발생 가능성 증가, 모델 사이즈의 증가 (많은 메모리 차지)
    
    - Small value
    
        - pros : 희소 문제의 발생 가능성 감소
    
        - cons : 낮은 정확도
    
    - Optimal value : $n \leq 5$

- 언어 모델 학습에 사용된 데이터셋으로부터 오는 한계
    
    - 특정 분야에 대한 단어가 많은 데이터를 이용해 언어 모델을 학습시켰다면, 다른 분야에 대한 단어를 예측하는 데 있어서는 높은 성능을 나타내지 못함.

## 2.5. 한국어 언어 모델의 특징

- 일반적으로 영어에 비해 한국어는 언어 모델을 통해 다루기 어려움

1. 어순이 중요하지 않음

    - 어순을 지키지 않은 문장 역시 의미상으로는 틀리지 않은 경우가 많음 ⇒ 단어 예측이 어려워짐

2. 교착어

    - ex) 조사
        - 사람 ⇒ 사람이, 사람을, 사람과, 사람보다, 사람처럼...

    - 토큰화를 통해 조사를 분리해야 함

3. 띄어쓰기

    - 띄어쓰기를 지키지 않아도 의미상으로 동일한 경우가 존재

    - 띄어쓰기 규칙의 까다로움 ⇒ 제대로 지켜지지 않는 경우가 대다수

    - 토큰화를 통해 해당 문제를 해결한 후 언어 모델을 구축해야 함

## 2.6. Perplexity

- 대표적인 언어 모델 평가 방법

    - 정성적으로, "헷갈리는 정도" 를 의미

    - 수치가 낮을수록 성능이 높음을 의미

- 단어의 개수가 $N$개인 문장 $W$가 존재할 때, 두 언어 모델의 성능을 비교할 수 있는 정량적인 지표 

    - "특정 언어 모델이, 특정 시점에서 **평균적으로 몇 개의 선택지를 두고 고민하고 있는지**"
    
    - ex) PPL = 10 인 경우 : 평균적으로 10개의 단어 중 하나를 정답으로 가지고 있다는 의미

- *Definition* of PPL
    
    $$\text{PPL}(W) = (P(w_1, w_2, \ \cdots, w_N))^{-1/N}=\sqrt[N]{\frac{1}{\prod\limits_{i=1}^{N}P(w_{i}| w_{1}, w_{2}, ... , w_{i-1})}}$$ 

- n-gram 모델에서 PPL을 계산하면?

    $$PPL(W)=\sqrt[N]{\frac{1}{\prod\limits_{i=1}^{N}P(w_{i}| w_{i-1}, w_{i-2}, \ \cdots , w_{i-n+1})}}$$ 

## 2.7. 단어 가방 모델 (Bag of Words, BoW)

- BOW : 주어진 텍스트 내 **어떠한 단어가 각각 몇 번씩 등장**하는지 저장한 모델

    - 단어의 출현 빈도에 집중

        <p align="center"><img src="https://remnote-user-data.s3.amazonaws.com/u_VE3qToQt3W5jrYPDSN4i2x34BpM24i2DX2tzA47lGY-1vpVAV0R2DII8ZKwukyoTyxt3oflRVIB74R5WuQsIs_W_bsmjlMXGZ0K0XSZLO5QkvdqPLPo8wqVZddTWvP.png" width="580"></p>

- 각 단어의 수를 나타내는 벡터 생성

- BOW의 특징 (2가지) 
    1. 단어 간 순서와 문법 무시

    2. 단어의 개수만을 고려

- TF / IDF

    - TF : Term-frequency, 특정 단어가 문서에서 등장하는 빈도

    - IDF : nverse term-frequency, 역 문서 빈도
    
    - 자세한 내용 및 구현은 NLP 실습 4 참고

- Example of BOW

    - **소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.**  
        - vocabulary : `{'소비자': 0, '는': 1, '주로': 2, '소비': 3, '하는': 4, '상품': 5, '을': 6, '기준': 7, '으로': 8, '물가상승률': 9, '느낀다': 10}`

        - bag of words vector : `[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]  `
        
        ⇒ ` {(소비자, 1), (는, 1), (주로, 1), $\cdots$, (을, 2), (기준, 1) ... } `

    - **정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.**  

        - vocabulary : `{'정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9, '는': 10, '주로': 11, '소비': 12, '상품': 13, '을': 14, '기준': 15, '으로': 16, '느낀다': 17}`

        - bag of words vector : `[1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]  `

        ⇒ `{(정부, 1), (가, 2), (발표, 1), (하는, 2), (물가상승률, 3), ... }`

## Reference
- 인공지능 교육단체 OUTTA 와 함께 하는! 자연어 처리 첫 단추 끼우기, OUTTA, 2022

- [Won Joon Yoo, Introduction to Deep Learning for Natural Language Processing, Wikidocs](https://wikidocs.net/21695) 

> 별도의 출처가 표기되지 않은 Figure의 경우 《인공지능 교육단체 OUTTA 와 함께 하는! 머신러닝 첫 단추 끼우기》에서 가져왔거나 원본을 기반으로 재구성 된 Figure 입니다.
