# AI_X_DEEPLEARNING

### Title: Global YouTube Statistics 2023

### Members:
          김재윤, 수학과, flhbigfamily7@gmail.com
          최성원, 기계공학부 davdev3411@gmail.com
          임정성, 신소재공학부 wiuand@gmail.com


## Index
####           I. Proposal
####           II. Datasets
####           III. Methodology
####           IV. Evaluation & Analysis
####           V. Related Works
####           VI. Conclusion: Discussion

## I. Proposal
+ Motivation
  대부분의 사람들이 스마트폰으로 동영상을 보는 시대가 되면서 유튜브와 방송 프로그럄의 경계가 흐려지고 있습니다. 또한 유튜버의 방송 프로그램 출연이 이제는 자연스럽고, 연예인의 유튜브 출연도 홍보를 위한 필수코스가 되었습니다. 많은 방송인들과 소속사들이 유튜브를 방송송출 플랫폼으로 정하는 상황에서, 유튜브상의 여러 지표들(구독자수, 월수입, 카테고리, 조회수 등)간의 관계를 파악하는것은 저희에게 통찰력을주고 도움이 될거라 생각합니다. 우리는 2023년 현재 상위 크리에이터의 구독자 수, 동영상 조회수, 업로드 빈도, 출신 국가, 수입 등이 나타나있는 데이터를 분석하고, 시각화해, 인기 유튜브채널의 성공요인과 현재 인기있는 카테고리 등을 알아보고, 구독자수와 조회수에 따른 수익예측을 해볼 계획입니다.
+ What do you want to see at the end?
  머신러닝의 여러모델을 통해 데이터를 학습시켜, 현재 유튜브 트랜드와 무엇이 성공한 유튜브채널의 요인인지를 알아보고 또한 수입과 구독자수등의 상관관계를 살펴보고 싶습니다.
  
## II. Datasets
+ kaggle에서 2023년 현재까지 구독자수 Top 995 유튜브 채널들의 정보를 모아둔 데이터셋을 활용하였습니다.
+ https://www.kaggle.com/datasets/nelgiriyewithana/global-youtube-statistics-2023/
+ feature들의 종류는 총 28개로 아래와 같습니다.

          rank                                      : 구독자 수에 따른 YouTube 채널의 위치
          Youtuber                                  : YouTube 채널 이름
          subscribers                               : 채널 구독자 수
          video views                               : 채널에 있는 모든 동영상의 총 조회수
          category                                  : 채널의 카테고리
          Title                                     : YouTube 채널의 제목
          uploads                                   : 채널에 업로드된 총 동영상 수
          Country                                   : YouTuber의 국가
          Abbreviation                              : 국가의 약어
          channel_type                              : YouTube 채널 유형(예: 개인, 브랜드)
          video_views_rank                          : 총 동영상 조회수를 기준으로 한 채널 순위
          country_rank                              : 해당 국가의 구독자 수를 기준으로 한 채널 순위
          channel_type_rank                         : 유형(개인 또는 브랜드)에 따른 채널 순위
          video_views_for_the_last_30_days          : 지난 30일 동안의 총 동영상 조회수
          lowest_monthly_earnings                   : 채널의 가장 낮은 예상 월 수입
          highest_monthly_earnings                  : 채널의 가장 높은 예상 월 수입
          lowest_yearly_earnings                    : 채널의 가장 낮은 예상 연간 수입
          highest_yearly_earnings                   : 채널의 연간 예상 수입 중 가장 높은 수익
          subscribers_for_last_30_days              : 지난 30일 동안 얻은 신규 구독자 수
          created_year                              : YouTube 채널이 만들어진 연도
          created_month                             : YouTube 채널을 만든 월
          created_date                              : YouTube 채널의 정확한 생성 날짜
          Gross tertiary education enrollment (%)   : 해당 국가의 고등 교육에 등록한 인구의 비율
          Population                                : 국가의 총 인구
          Unemployment rate                         : 국가의 실업률
          Urban_population                          : 도시 지역에 거주하는 인구의 비율
          Latitude                                  : 국가 위치의 위도 좌표
          Longitude                                 : 국가 위치의 경도 좌표

## III. Methodology
우리는 연속된 값인 '연수입'을 예측할 것 이므로 선형회귀모델을 사용할것입니다. 선형회귀는 종속변수 y와 한 개 이상의 독립변수 X와의 선형 상관 관계를 관찰해서 독립변수를 통해 종속변수를 예측하는 방법입니다. 독립변수가 한개면 단순 선형 회귀, 두개 이상이면 다중 선형 회귀라고 합니다.
+ 먼저 데이터의 결측치를 제거하는 전처리과정을 거치고, 변수들과의 상관관계를 히트맵을 통해 확인해볼겁니다.
+ 다음으로 구독자수 한개의 feature만 가지고(단순선형회귀) 선형회귀모델중 하나인 Linear regression을 이용해 연수입을 예측해볼 것입니다.
+ 마지막으로 모든 변수들을 다 이용해서(다중선형회귀) 똑같이 Linear regression을 이용해 연수입을 예측해보고, 선형회귀모델중 또다른 종류인 Ridge regression을 이용해서 연수입을 예측해 두 모델의 정확성을 비교해볼 것입니다.

## IV. Evaluation & Analysis
+ 먼저 pandas 라이브러리를 통해 데이터셋을 불러와 데이터프레임을 만들어주고 df.info()를 이용해 data의 정보를 확인해 봅니다.
```python
import pandas as pd

df = pd.read_csv("Global YouTube Statistics.csv", encoding='unicode_escape')
df.info() #995개의 열과 28개의 feature들이 있는것을 확인가능
df
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/a542073e-d2f6-4914-ad5c-6a69de2fb2a9.png" width="400" height="400"/>
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/429c6ec1-f154-441a-bb74-6c84e6cb55d5" width="700" height="400"/>

+ 데이터의 행(채널)은 총 955개가 있고, 열(특징)은 28개가 있는것을 확인할수 있습니다. 또 구독자수가 전부 천만명이상에 2억명까지도 있는 것을 볼수 있습니다.
+ 다음으로 결측치를 제거하기위해 행별로 결측치의 수를 확인해 보겠습니다.
```python
df.isnull().sum()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/b4683196-6dfa-4162-90a9-d1e9813e429d" width="400" height="400"/>

+ 확인해보면 subscribers_for_last_30_days란 행이 337개나 결측치를 갖고있어 모든 결측치를 제거해버리면 너무많은 데이터가 사라지므로 이 열은 데이터에서 삭제 할것입니다.
```python
df = df.drop('subscribers_for_last_30_days', axis = 1)
```
+ 다음으로 모든 결측치를 데이터에서 지워주고, 데이터를 확인해봅니다.
```python
df = df.dropna(axis = 0)
df.info()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/279bb520-ef81-437f-a54e-d74845baa419" width="400" height="400"/>

+ 결측치를 제거후 총 808개의 데이터가 남아있는것을 확인할수 있습니다.
+ 다음으로 유튜브 채널의 성공요인을 분석해 보겠습니다. 먼저 데이터에서 나라종류의 갯수를 확인해 봅니다.

```python
len(df['Country'].unique())
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/6e4fc7fe-dcf7-462d-9e03-62dc0c813f61" width="150" height="20"/>

+ 총 47개의 나라가 있는것을 확인할수 있고, 이제 matplotlib, seaborn 라이브러리를 통해 top1000 유튜버들의 나라의 분포를 히스토그램으로 시각화해 봅시다.
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (15, 7))
ax = sns.countplot(data = df, x = 'Country', order = df['Country'].value_counts().head(10).index)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 5, height, ha = 'center', size = 12)
ax.set_ylim(0, 350)
plt.show()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/5f6d67ea-e26e-4fd0-b270-9e07aeba9571" width="800" height="400"/>

+ top1000 유튜브 채널중 미국이 압도적으로 30퍼이상을 차지하고있고, 인도 브라질이 따라오는 것을 확인할수 있습니다. 그리고 공동9위로 한국이 있는것도 볼수 있습니다.
+ 다음으로 유튜브 채널의 카테고리를 히스토그램으로 나타내겠습니다.
```python
plt.figure(figsize = (17, 7))
ax = sns.countplot(data = df, x = 'category', order = df['category'].value_counts().head(10).index)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 2, height, ha = 'center', size = 12)
ax.set_ylim(0, 250)
plt.show()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/cc925480-cf8c-4b70-af1d-3157710b98fe" width="800" height="400"/>

+ 예능, 오락채널인 Entertainment가 208개로 1등, 다음 근소한 차이로 Music카테고리가 2등 이였습니다.
+ countplot으로도 확인해보면,
```python
plt.figure(figsize = (7, 7))
explode = [0.05 for i in range(len(df['category'].unique()))]
df['category'].value_counts().plot.pie(ylabel='', title = 'category',
                                            autopct = '%1.1f%%', fontsize = 15)
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/8c63d98f-4b78-41ad-8501-7289653e4897" width="800" height="600"/>

+ top 800유튜브 채널의 채널유형의 70퍼 이상이 Entertainment, Music, People&Blogs, Gaming 카테고리임을 확인할수 있습니다. 유튜브로 성공하려면 예능, 음악, 게임방송을 하는게 확률이 높을것 같습니다...

+ 다음으로 탑 유튜브 채널중 한국채널만 골라내서 카테고리를 살펴봅시다.
```python
df_k = df[df['Country'] == 'South Korea']
df_k
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/a948860b-37bd-42f3-a8d8-82a4d86e60b0)" width="600" height="400"/>

+ 언뜻봐도 블랙핑크, 방탄, 하이브 등등 k-pop아이돌 관련 채널들이 눈에 띕니다. 이제 이 카테고리들을 히스토그램으로 만들어봅시다.
```python
ax = sns.countplot(data = df_k, x = 'category', order = df_k['category'].value_counts().head().index)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, height, ha = 'center', size = 12)
ax.set_ylim(0, 8)
plt.show()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/3b0f1afb-1afa-41f4-85d9-00b82b85c652" width="400" height="400"/>

+ Entertainment쪽에도 k-pop 음악방송이 포함되있어, 총 15개 채널중 8개가 k-pop아이돌 관련 채널입니다. 역시 아이돌 강국...!!
+ 
<img src="" width="400" height="400"/>
