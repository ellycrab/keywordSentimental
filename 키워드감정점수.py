from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from vader import *
import nltk
nltk.download('vader_lexicon')

finbiz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN','AMD','FB']

news_tables = {}
for ticker in tickers:
    url = finbiz_url + ticker

    req = Request(url=url, headers={'user-agent':'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    # 위의 news_table에서 id에 해당하는 것들을 리스트로
    # news_tables의 빈 공간에 넣습니다.
    news_tables[ticker] = news_table

parsed_data = [] 
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.strip().split(' ')
        date = date_data[0]
        if len(date_data) > 2:
            time = date_data[2]
        else:
            time = ""
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns=['ticker','date','time','title'])
print(df)

#VADER Sentiment 모듈 초기화
analyzer = SentimentIntensityAnalyzer()
# 감정 점수 계산
text = "I don't think Apple is a good company. I think they will do poorly this quarter."
sentiment_scores = analyzer.polarity_scores(text)
# 감정 점수 출력
print(sentiment_scores)
#lambda함수를 사용하여 f라는 이름으로 정의한 함수는 title
#에대하여 감성분석을 통해 감성점수를 계산하고
#그 중에 compound점수를 반환한다.
f = lambda title: analyzer.polarity_scores(title)['compound']

# apply를 사용하여 'title' 컬럼의 각 값에 함수 f를 적용
df['compound'] = df['title'].apply(f)
df

# 'compound' 컬럼의 값을 숫자로 변환
#errors='coerce'를 사용하여 결측치 오류 발생막음
# 'compound' 컬럼의 값을 숫자로 변환
df['compound'] = pd.to_numeric(df['compound'], errors='coerce')

# 날짜 컬럼을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# mean_df 생성
mean_df = df.groupby(['ticker', 'date'])['compound'].mean()


# mean_df를 ticker와 date를 인덱스로 변환하여 다중 인덱스 데이터프레임으로 만듦
mean_df_multiindex = mean_df.unstack('ticker')

# 그래프 크기 설정
plt.figure(figsize=(10, 8))

# 그래프 그리기
mean_df_multiindex.plot(kind='bar', ax=plt.gca())  # plt.gca()는 현재 AxesSubplot을 가져옵니다.

# 그래프 제목 및 라벨 설정
plt.title('Mean Sentiment Score Over Time')
plt.xlabel('Date')
plt.ylabel('Mean Sentiment Score')

# x축 날짜 라벨을 45도 회전해서 표시
plt.xticks(rotation=45)

# 그래프 표시
plt.tight_layout()
plt.show()
