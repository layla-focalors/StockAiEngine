# StockAiEngine

이 레포지트리는 SqlLite를 활용한 데이터베이스 기반의 주식 분석 데이터 작업을 다룹니다.  

### 개발 예정 과정
1. 파이썬 기반의 주식 예측 시스템 개발 - 1차
2. Rust 기반으로 포팅 & 재작성  
3. 웹 기반 ui 추가 - Next.js 
4. 서비스 배포 - AWS, Azure, GCP // 유료 요금제 추가  
5. 자동운영 추가 - 키움 API

### 예측 모델
LSTM / RNN
TYPE : Pytorch 

### 추가 개발
LSTM - 만약 사전 학습 모델이 있다면 해당 모델의 가중치 불러오기
없다면 학습 과정부터 진행  
실행 시 가중치 로드 후 분석할것인지 질문  
분석 종료 후 OpenAI ChatGPT / Ollama / OndeviceLLM 호출 후 분석 리포트 작성 후 안내