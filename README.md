# Chatbot by Hotsix
<img width="500" alt="IMG_7677" src="https://user-images.githubusercontent.com/96299403/219873027-111fe19b-52a7-4955-afa5-3f7ff9f2d972.JPG">

## Introduction
필요한 기술을 어떤 기업이 보유하고 있는지 찾는 데 요구되는 시간적•비용적 부담을 줄이기 위해 개발한 기술 키워드 기반 기업 검색 챗봇입니다.\
질문에 포함된 기술 키워드를 기반으로 관련 기업을 검색하고, 해당 기업들에 대한 정보, 유사한 기업들을 빠르게 조회할 수 있습니다.

## Overview
<img width="500" src="https://user-images.githubusercontent.com/96299403/219872851-e7f55c6f-c64b-4d73-9e77-37c89162585a.PNG">
https://youtu.be/Iezcii3gfkU

### Functions
1. 기술 기반 기업 추천
- 챗봇 메뉴에서 기술 키워드를 사용하여 질문하면, 해당 기술 키워드를 포함하는 기업 중 가장 관련 있는 기업을 검색 결과로 제시합니다.
2. 유사 기업 추천
- 검색한 기업과 유사한 기술 키워드를 보유한 기업들 중 유사도가 가장 높은 상위 5개 기업을 검색 결과로 제시합니다.
3. 기업 주요 정보 제공
- 검색한 기업에 관한 최신 기사와 해당 기사들의 감성 분석 결과를 요약하여 제시합니다.
- 검색한 기업의 주요 재무 정보를 시각화하여 제시합니다.

## Installation
```
git clone https://github.com/kpmg-hotsix/kpmg_hotsix_chatbot.git
pip install -r requirements.txt
```

## Usage
1. 다음 커맨드로 애플리케이션을 실행합니다.
```
python app.py
```
2. 챗봇 영역에서 질문을 입력합니다.
- 음성 인식과 챗봇 솔루션을 갖춘 기업을 알려줘.
- OO기업의 최근 이슈를 알려줘.
3. 질문에 대해 챗봇의 답변을 받고, 오른쪽 화면에서 상세 정보를 확인할 수 있습니다.

## Package Structure
```
- data      # 프로토타입 앱 구동에 필요한 데이터  
- process   # 데이터 수집 및 NLP 모델 학습에 사용된 코드
```
