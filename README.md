# AI_X_DEEPLEARNING

### Title: Top 1000 유튜버들을 분석하고 연수입 예측해보기

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
  많은 사람들이 스마트폰으로 동영상을 보는 시대가 되면서 유튜브와 방송 프로그램의 경계가 흐려지고 있습니다. 또한 유튜버의 방송 프로그램 출연이 이제는 자연스럽고, 연예인의 유튜브 출연도 홍보를 위한 필수코스가 되었습니다. 또한 유튜브를 통해 정보를 얻기가 쉬워지고있고 많은 방송프로그램들이 유튜브를 방송송출 플랫폼으로 정하는  유튜브의 중요성이 부각되는 상황에서, 유튜브상의 여러 지표들(구독자수, 월수입, 카테고리, 조회수 등)간의 관계를 파악하는것은 저희에게 통찰력을주고 도움이 될거라 생각합니다. 우리는 2023년 현재 상위 크리에이터들의 데이터를 분석하고 시각화해, 인기 유튜브채널의 성공요인과 현재 인기있는 카테고리 등을 알아보고, 구독자수와 조회수에 따른 수익예측을 해볼 계획입니다.
+ What do you want to see at the end?
  나라마다 얼마나 상위 크리에터들이 있는지 확인해보고 현재 유튜브 트랜드를 살펴볼것 입니다. 또한 연수입과 나머지 요인들의 상관관계를 살펴보고, 머신러닝의 선형회귀모델들을 통해 데이터를 학습시켜 독립변수들을 통해 연수입을 예측해보고, 두 선형회귀 모델간의 차이를 살펴볼 것입니다.
  
## II. Datasets
+ kaggle에서 2023년 현재까지 구독자수 Top 995개의 유튜브 채널들의 정보를 모아둔 데이터셋을 활용하였습니다.
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
+ 먼저 데이터의 결측치를 제거하는 전처리과정을 거치고 탑 유튜버들의 채널유형(카테고리)와 나라분포를 확인해 볼겁니다. 또한 변수들간의 상관관계를 히트맵을 통해 확인해 보겠습니다.
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
+ 다음으로 연수입을 예측해 볼것이니, 연 최고수입과 연최저수입의 평균을 내서 연수입column을 만들어 봅시다.
```python
df['yearly_earnings'] = (df['highest_yearly_earnings'] + df['lowest_yearly_earnings']) / 2
df[['yearly_earnings', 'highest_yearly_earnings', 'lowest_yearly_earnings']]
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/6330e012-0ff8-40ed-ad32-5e7b6d46477b" width="400" height="400"/>

+ 'yearly_earnings'라는 새 컬럼이 생성된것을 확인할수 있다.
+ 이제 선형회귀분석을 할것이기 때문에 수치형 데이터컬럼만 남겨줍니다. 또한 나라의 실업률, 고등 교육률, 국가의 위도경도등은 우리가 유튜브채널을 운영한다고 할때 조절할수 없는 요인들이므로 이것들은 빼고 진행하겠습니다.
+ 수치형데이터들간의 상관관계 heatmap을 표시합니다.
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/a0dc88be-6db3-4649-b9db-8e0358543543" width="400" height="400"/>

+ 숫자가 1에 가까울수록 양의 상관관계(커지면 커진다), -1에 가까울수록 음의 상관관계(커지면 작아진다)가 있습니다. 히트맵을 보면 연수입은 구독자수와 동영상조회수랑 큰 양의상관관계가 있음을 알수 있고, 업로드수, 채널생성연도, 나라인구수와는 큰 상관관계가 없는것을 알수 있습니다.
+ 일단은 높은 상관관계를 가지고있는 특징중 하나인 구독자수를 통해 연수입을 예측해봅시다.
+ 먼저 필요한 라이브러리들을 불러오고 구독자수를 독립변수, 연수입을 종속변수로 둡니다. 다음으로 훈련용 데이터와 테스트용 데이터를 8:2비율로 나눠줍니다.
```python
import numpy as np
from sklearn.model_selection import train_test_split

X_sub = df_lr['subscribers']
y_earn = df_lr['yearly_earnings']
X_sub = np.array(X_sub).reshape(-1, 1)

#train_test_split를 이용해 훈련용, 테스트용 데이터를 8:2로 나눠준다.
X_train, X_test, y_train, y_test = \
train_test_split(X_sub, y_earn,
                 test_size = 0.2,
                 random_state = 77)
```
+ 다음으로 사이킷런에서 LinearRegression 모듈을 불러오고, 훈련용데이터를 학습시켜줍니다.
```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_sub, y_earn)
```
+ 다음으로 mean_squared_error 모듈을 통해 예측의 정확성을 살펴봅시다. 이때 mse가 작을수록 성능이 좋은것을 나타냅니다.
+ rmse는 mse에 루트를 씌운것
```python
from sklearn.metrics import mean_squared_error

y_train_predict = reg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))

print("For training set")
print("RMSE is ", rmse)
```
out : 

For training set

RMSE is  6016174.478731269
+ mse가 엄청 크게 나오는데 연수입 단위가 크기때문에 크게 측정된 점도 있습니다.
+ 다음으로 test셋에 대해 mse를 측정해 봅시다.
```python
y_test_predict = reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

print("For test set")
print("RMSE is ", rmse)
```
out :

For test set

RMSE is  5372666.424711112
+ 훈련용 데이터셋을 통해 한 예측보다는 성능이 좋은것을 알수 있습니다.
+ 이제 scattereplot을 통해 실제값과 예측값을 비교해봅시다.
```python
prediction_space = np.linspace(min(X_sub), max(X_sub))
plt.scatter(X_sub, y_earn)
plt.plot(prediction_space, reg.predict(prediction_space),
         color = 'black', linewidth = 3)
plt.ylabel('yearly earning/1000($)')
plt.xlabel('number of subs')
plt.show()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/f39f347a-affd-4054-b629-742e2689747d" width="400" height="400"/>

+ 데이터들이 5천만 구독자 이하에 거의다 몰려있어서 제대로 확인할수 없습니다. 따라서 5천만 이하의 채널들만 확인해 봅시다.

```python
df_lr5 = df_lr[df_lr['subscribers'] < 50000000]

X_sub = df_lr5['subscribers']
y_earn = df_lr5['yearly_earnings']
X_sub = np.array(X_sub).reshape(-1, 1)


X_train, X_test, y_train, y_test = \
train_test_split(X_sub, y_earn,
                 test_size = 0.3,
                 random_state = 77)

reg = LinearRegression()
reg.fit(X_sub, y_earn)

prediction_space = np.linspace(min(X_sub), max(X_sub))
plt.scatter(X_sub, y_earn)
plt.plot(prediction_space, reg.predict(prediction_space),
         color = 'black', linewidth = 3)
plt.ylabel('yearly earning/1000($)')
plt.xlabel('number of subs')
plt.show()
```
<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/9ee8749f-f4dc-45c6-aede-04adb0eeb575" width="400" height="400"/>

+ 몇가지 튀는점들이 있지만 어느정도 예측선을 따라가는 경향이 있습니다.
+ 이제 만든 모델을 가지고 구독자수를 통해 연수입을 예측해봅시다!
```python
print(reg.predict([[1000000]]))
print(reg.predict([[500000]]))
```

out:

[149434.07198932]

[63552.12820991]

+ 구독자수가 100만명일때는 연수입 예측값은 2억원정도이고, 50만명 일때는 8천만원 정도가 예측됩니다. 실제로는 더 높을것같긴 합니다..
+ 이제 구독자수 뿐만 아니라 동영상조회수, 업로드수, 인구 등 까지 독립변수로 두고 다변수 회귀분석을 해봅시다.

```python
X = df_lr.drop('yearly_earnings', axis = 1)#axis가0일경우 index, 1일 경우 colimns제거
y = df_lr['yearly_earnings']

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size = 0.2,
                 random_state = 77)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
```

```python
y_train_predict = reg_all.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))

print("For training set")
print("RMSE is ", rmse)
```

out:

For training set

RMSE is  5243563.199390768

```python
y_test_predict = reg_all.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

print("For test set")
print("RMSE is ", rmse)
```

out:

For test set

RMSE is  4316246.653554859

+ 구독자수만을 통해 예측할때보다 훈련용, 테스트셋에서 모두 성능이 좋아진것을 확인할수 있습니다.
+ 이제 scatterplot을 통해 실제 연수입과 예측연수입을 비교해 봅시다.
+ 
```python
plt.scatter(y_test, y_test_predict)
plt.xlim([0, 10000000])
plt.ylim([0, 10000000])
plt.xlabel("Actual yearly earnings ($1000)")
plt.ylabel("Predicted yearly earnings ($1000")
plt.title("Actual earnings vs Predicted earnings")
plt.plot([0, 40000000], [0, 40000000], 'r')
```

<<img src="https://github.com/jayjayppark/aix_deep_learning/assets/150012836/fdce77c8-7b11-4b9c-822d-b4ead170e332" width="400" height="400"/>img src="" width="400" height="400"/>

+ 가운데 빨간줄에 데이터들이 가까울수록 정확한건데, 튀는 점들이 눈에 띕니다. 하지만 대부분 예측의 경향성과 맞습니다.
+ 다음으로 선형회귀분석의 다른종류인 ridge regression을 이용해 예측해봅시다.
+ ridge regression은 훈련용 데이터에 과적합이 안되도록 특화된 방법입니다.

```python
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train,y_train)
y_pred = ridge.predict(X_test)

y_train_predict = ridge.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))

print("For training set")
print("RMSE is ", rmse)

y_test_predict = ridge.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

print("For test set")
print("RMSE is ", rmse)
```

out:

For training set

RMSE is  5243563.199976563

For test set

RMSE is  4316255.477563959

+ mse를 확인해보니 linear regression을 적용했을때랑 거의 차이가없고, test셋에서는 아주 조금 성능이 안좋아졌습니다.
+ 이 데이터에서는 linear regression이나 ridge regression 어느것을 사용해도 차이가 없음을 확인할수 있었습니다.

## V. Related Works

Tools, libraries:

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html#pandas.DataFrame.plot

blogs, or any documentation:

https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80

https://www.kaggle.com/code/zabihullah18/global-countries-eda

https://www.appier.com/ko-kr/blog/5-types-of-regression-analysis-and-when-to-use-them

https://signature95.tistory.com/46

## VI. Conclusion: Discussion

+ 탑 유튜버들을 제일 많이보유한나라는 역시 유튜브의 시초인 미국이고, 인도, 브라질, 인도네시아 등등 인구가 많은 국가들이 따라오고 있었습니다. 한국은 다른 top 10 국가에 비해 인구수가 적음에도 9위에 있는것을 확인할수 있었습니다.
+ 탑 유튜브 채널들의 유형을 관찰해보니, 현재 인기있는 카테고리는 예능, 음악, VLOG, 게임 정도로 확인할수 있었습니다.
+ 히트맵을 통해 연수입과 관련성이 큰 요인들을 알아보니 구독자수와 동영상조회수가 연수입과 가장 큰 연관이 있음을 확인 가능했습니다.
+ 구독자수만을 독립변수로 놓고 Scikit-Learn의 linear regression을 이용해 단순선형회귀분석을 해보니 구독자 100만명을 input으로 넣었을때 예측 연수입은 2억, 구독자수 50만명일때는 예측 연수입이 8천만원 정도가 나온다고 예측되었습니다.
+ 다음으로 같은 linear regression모델에 독립변수를 구독자수, 동영상조회수, 업로드수, 인구 모두로 설정하고, 연수입을 예측하는 다중선형회귀분석을 해보니 구독자수만을 독립변수로 넣을때보다 성능이 좀더 좋아진것을 확인가능했습니다.
+ 다음으로 선형회귀의 다른종류이고 과적합방지에 더 특화된 모델인 ridge regression을 이용해 다중선형회귀분석을 해보니, linear regression보다 test셋에서 좀더 안좋은 성능을 확인할수 있었지만 사실상 큰 차이가 없었습니다. 따라서 이 데이터셋에선 linear regression이나 ridge regression 뭘 써도 상관없다는 결론을 냈습니다.

## Youtube Link: 

### 김재윤 : YouTube recording, Blog Processing
### 최성원 : Code Implementation, Visualization
### 임정성 : Data Preprocessing, Analysis
