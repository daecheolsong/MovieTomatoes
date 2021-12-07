
# MovieTomatoes Ver 0.1
# 내용:
#   1) 네이버 영화에서 영화 리뷰 수집
#   2) 수집 된 리뷰 MongoDB에 저장
#   3) MongoDB 에서 수집 된 데이터 불러옴
#   4) 인공지능에 사용할 수 있게 전처리
#   5) 전처리 된 데이터를 활용하여 인공지능 분석 시작
#   6) 분석결과를 시각화

# 만든이 : daecheolsong
# 일자 : 2021. 11. 09
import math

import numpy as np
import requests
from bs4 import BeautifulSoup
import webcrawl.WebCrawlService as wcs
import model.MongoDAO as mongo
from konlpy.tag import Okt
import tensorflow as tf
##########################
# 1. 데이터 수집 및 저장    #
##########################

movie_code = '209496'  # 네이버 영화 code

'''
 review 수집
 제목, 리뷰, 점수 , 작성자, 날짜
'''


# 1.제목 수집

title = wcs.get_movie_title(movie_code)
print(title)

# 2.전체 페이지수 계산

pages = wcs.calc_pages(movie_code)
print(pages)

# 3. 리뷰 수집

# wcs.get_reviews(title,movie_code,pages)

########################
# 2. 인공지능 분석        #
########################

review_list = mongo.get_reviews()
#print(review_list[0])
#print(review_list[1])
#print(review_list[2])
#print(len(review_list))


########################
# 3. 분석결과 시각화      #
########################

def read_data(file_name):
    words_data = []
    with open(file_name,'r',encoding='UTF-8') as f:
        while True:
            line = f.readline()[:-1]
            if not line: break
            words_data.append(line)

    return words_data

selected_words = read_data('./ai/selectword.txt')


# 예측할 데이터의 전처리를 진행할 메서드
okt = Okt()
def tokenize(doc):

    # norm 은 정규화, stem 은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


# 예측할 데이터의 벡터화를 진행할 메서드
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


# 학습 된 인공지능 모델(AI) 불러오기
model = tf.keras.models.load_model('./ai/my_model.h5')
print('model(type):', type(model))


# 인공지능 모델로 긍부정 예측하는 메서드
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(review)
    data = np.expand_dims(np.asarray(tf).astype('float32'),axis=0)
    predict_score = float(model.predict(data))

    if(predict_score > 0.5):
        print('[{}] => {:.2f}% 확률로 긍정리뷰 예상'.format(review,predict_score*100))
    else:
        print('[{}] => {:.2f}% 확률로 부정리뷰 예상'.format(review,(1-predict_score)*100))

predict_pos_neg(review_list[0][1])