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