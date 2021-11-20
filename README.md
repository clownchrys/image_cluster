# Introduction
1. 본 레포지토리는 재직중에 개발했던 유사 이미지 추천을 위한 image clustering 에 대해 기록하기 위해 작성됨
2. 코드베이스가 남아있지 않아 실제 트레이닝 코드는 포함되지 않음
3. 서버 사이드의 데이터 파이프 라인은 아래에 간략하게 기술함

# Problem
- 배경: 매일 10만개 이상의 새 상품을 크롤링하여 판매하는 이커머스 (총 800만개 이상의 상품 보유)  
- 목적: 고객의 Interaction 에 따라 유사한 외관의 상품에 대한 추천기능을 제공하려고 하였음  

# Structure
1. Featurize: CNN 모델을 통해 Feature map 생성
2. Clusterize: KMeans를 통해 각 카테고리의 Feature를 군집화
3. Recommend
	- Interaction 한 상품의 군집에 속해 있는 상품을 추천
	- 군집 내의 Vector 위치와 가까운 상품을 우선 추천
4. Revalidate: 새로 들어오는 상품은 Feature map 으로 만들어서, 현재의 centroid 와 pca 를 통해 바로 군집화
	- airflow 의 crawling dag 의 task 를 추가하였음
5. Re-Clusterize: 일정 주기로 centroid 와 pca 를 다시 계산함
	- jenkins 를 통해서 주기적으로 실행됨

# Description
- 왜 KMeans 를 PyTorch 를 통해 재구현하였는가?
	- sklearn: 라이브러리가 GPU 로 실행되지 않으므로, 대용량의 이미지셋을 주기적으로 학습하기 어려움
	- 따라서, torch 를 통해 GPU 를 가용하여 주기적으로 학습함

# Example
<img src="https://i.imgur.com/nuVHOFW.png">
<img src="https://i.imgur.com/NlkTwyx.png">
<img src="https://i.imgur.com/7uZ6lcn.png">
<img src="https://i.imgur.com/sC2TpKy.png">
<img src="https://i.imgur.com/VjWK5zw.png">
