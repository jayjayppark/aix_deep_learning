# AI_X_DEEPLEARNING

### Title: Top 1000 유튜버들을 분석하고 연수입 예측해보기

### Members:
          김재윤, 수학과, flhbigfamily7@gmail.com
          최성원, 기계공학부 davdev3411@gmail.com
          임정섭, 신소재공학부 wiuand@gmail.com


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
          channel_type_rank                  섭 : Data Preprocessing, Analysis
