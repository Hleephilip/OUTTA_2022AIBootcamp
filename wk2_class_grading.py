#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle
import gzip
import numpy as np
import random
from wk2_ans import operation

def grade():
    with gzip.open('wk2_class_data.pickle','rb') as f:
        data = pickle.load(f)
    np.random.randint
    correct = True

    lst = range(500)
    test_cases = sorted(random.sample(lst, 100))
    op = operation()

    for i in range(100):
        op.arrayA = data["A"][test_cases[i]]
        op.arrayB = data["B"][test_cases[i]]
        ans_C = data["ANS"][test_cases[i]]
        ans_sum_value = data["SUMVALUE"][test_cases[i]]  
        C, sum_value = op.calculate()

        if (np.array_equal(C, ans_C) == False):
            print("\r", end = "")
            print(f"A ◎ B 행렬의 원소 값이 잘못되었습니다.")
            print()

            print("[행렬 A]")
            print(op.arrayA)
            print()

            print("[행렬 B]")
            print(op.arrayB)
            print()

            print("[출력 결과]")
            print(C)

            print()
            print("[정답]")
            print(ans_C)

            correct = False
            break


        if (C.shape != ans_C.shape): 
            print("\r", end = "")
            print(f"A ◎ B 행렬의 Size 가 잘못되었습니다.")
            print()

            print(f"A.shape : {op.arrayA.shape}")
            print(f"B.shape : {op.arrayB.shape}")
            print(f"출력 결과 : {C.shape}")
            print(f"정답 : {ans_C.shape}")

            correct = False
            break

        if (sum_value != ans_sum_value):
            print("\r", end = "")
            print(f"sum_value 값이 잘못되었습니다.", end = "")
            correct = False
            break

        print("\r", end = "")
        print(f"채점 진행 중... ({int(i)}%)", end = "")

    if (correct):
        print("\r", end = "")
        print(f"모든 Test Case를 통과하였습니다!", end = "")


# In[ ]:




