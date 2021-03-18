#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 공부시간 X와 성적 Y의 리스트 만들기
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#그래프로 표시
plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.show()

#리스트 -> 넘파이 배열
x_data = np.array(x)
y_data = np.array(y)

# 기울기 a, 절편 b 초기화
a = 0
b = 0

#학습률 설정
lr = 0.03

# 반복 횟수 설정
epochs = 2001

#경사 하강법 
for i in range(epochs):
    y_pred = a * x_data + b # y를 구하는 식
    error = y_data - y_pred # 오차 구하는 식

    # 오차 함수를 a로 편미분
    a_diff = -(2/len(x_data)) * sum(x_data * (error))
    # 오차 함수를 b로 편미분
    b_diff = -(2/len(x_data)) * sum(error)

    # 학습률을 곱해 기존의 a, b 값 업데이트
    a = a - lr * a_diff
    b = b - lr * b_diff

    if i % 100 == 0:
        print("epochs= %.f, 기울기= %04f, 절편= %04f" % (i, a, b))

# 기울기와 절편을 이용해 그래프 리프레쉬
y_pred =  a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()

# %%
