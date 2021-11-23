# ai > ModelLearning.py

# Generate A.I model
# >> Model: 긍부정 분석 모델(감성분석)
# >> Module: Tensorflow, Keras
# >> Dataset: Naver Sentiment Movie Corps (https://github.com/e9t/nsmc/)

###################
# Dataset Intro   #
###################

# 데이터셋: Naver Sentiment Movie Corps
# >> 네이버 영화 리뷰 중 영화당 100개의 리뷰를 모아
# >> 총 200,000 개의 리뷰( 훈련 15만개, 테스트 5만개)로
# >> 이루어져있고, 1~10점까지의 평점 중 중립적인 평점(5~8)은
# >> 제외하고 1~ 4 점을 긍정 9~10점을 부정으로 동일한 비율로
# >> 데이터에 포함시킴

'''

데이터는 id, document, label 세개의 열로 이루어져있음
id : 리뷰의 고유한 key값
document: 리뷰의 내용
label: 긍정(1) 인지 부정(0)인지 나타냄
       평점이 긍정(9~10), 부정(1~4) , 5~8은 제거

'''

import json
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
from pprint import pprint
from konlpy.tag import Okt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

#############
# File Open #
#############

# ~.txt 파일에서 데이터를 불러오는 method

def read_data(filename):

    with open(filename,'r',encoding='UTF-8') as f:
        data = [ line.split('\t') for line in f.read().splitlines()]
#       data = []
#       for line in f.read().splitlines():
#           data.append(line.split('\t'))

        data = data[1:] # 제목열 제외
    return data

# nsnc 데이터를 불러와서 python 변수에 담기
train_data = read_data('./dataset/ratings_train.txt') # 트레이닝 데이터 Open
test_data = read_data('./dataset/ratings_test.txt') # 테스트 데이터 Open
# / -> 하위폴더
# .. -> 상위폴더
# . -> 현재폴더


# 절대경로와 상대경로
# C:/pypy_workspace/MovieTomatoes
#                     ㄴ ai
#                       ㄴ dataset
#                           ㄴ ratings_text.txt
#                           ㄴ ratings_train.txt
#                       ㄴ ModelLearning.py
#                     ㄴ model
#                     ㄴ webcrawl
# ....


print(len(train_data))
print(train_data[0])

print(len(test_data))
print(test_data[0])



