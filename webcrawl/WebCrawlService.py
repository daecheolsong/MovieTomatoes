import math
import re
import requests
from bs4 import BeautifulSoup
from model.MongoDAO import add_review


def get_movie_title(movie_code):

    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text,'html.parser')

    title = doc.select('h3.h_movie a')[0].get_text()
    return title


def calc_pages(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text,'html.parser')

    all_count = doc.select('strong.total > em')[0].get_text().strip()
    numbers = re.sub(r'[^0-9]', '', all_count) # 정규식활용 => 0~9숫자 외의 값은 ''으로 제거
    pages = math.ceil(int(numbers) / 10)
    return pages


def get_reviews(title,movie_code,page):
    count = 0  # Total Review Count

    for page in range(1, page + 1):

        new_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={}&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}'.format(movie_code,
            page)
        result = requests.get(new_url)
        doc = BeautifulSoup(result.text, 'html.parser')
        review_list = doc.select('div.score_result ul li')

        for one in review_list:
            count += 1
            print('## USER  -> {} ####################################################'.format(count))

            # 평점 정보 수집
            score_list = one.select('div.star_score em')
            score_0 = score_list[0]
            score_text = score_0.get_text()

            # 리뷰 정보 수집
            review = one.select('div.score_reple p span')[-1].get_text().strip()

            # 작성자(닉네임) 정보 수집
            original_writer = one.select('div.score_reple dt em')[0].get_text().strip()

            idx_end = original_writer.find('(')  # (가 해당하는 idx위치 리턴
            # (xxx****) 제거
            writer = original_writer[0:idx_end]

            # 날짜 정보 수집
            original_date = one.select('div.score_reple dt em')[1].get_text()
            # yyyy.MM.dd 전처리 코드 작성
            idx_end = original_date.find(' ')
            date = original_date[0:idx_end]

            print(':: TITLE -> {}'.format(title))
            print(':: REVIEW -> {}'.format(review))
            print(':: WRITER -> {}'.format(writer))
            print(':: SCORE-> {}'.format(score_text))
            print(':: DATE -> {}'.format(date))

            data = {'title': title,
                    'score': score_text,
                    'review': review,
                    'writer': writer,
                    'date': date}
            # MongoDB에 Review 저장
            # Mongo dict type 으로 저장
            add_review(data)
