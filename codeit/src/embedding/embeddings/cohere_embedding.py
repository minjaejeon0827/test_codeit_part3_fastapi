"""
cohere_embedding.py - Cohere 임베딩

BaseEmbedding 클래스 상속받아 Cohere 임베딩 모델 사용. (embed-v4.0 등)
전략 패턴 (Strategy Pattern) 기반 클래스.

* Cohere 임베딩 강점:
    - RAG 특화 (검색·임베딩·리랭킹 전문 회사)
    - 100개+ 언어 지원 (한국어 공식 지원, RFP 문서에 적합)
    - 금융/의료/제조 산업 문서에 미세조정됨
    - MRL 기술로 차원 선택 가능 (256/512/1024/1536)
    - 무료 Trial 티어 제공

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding               (Strategy 인터페이스)
        ├── HFEmbedding         (로컬 GPU)
        ├── OpenAIEmbedding     (유료 API)
        ├── OllamaEmbedding     (로컬, 무료)
        ├── GeminiEmbedding     (무료 티어 API)
        └── CohereEmbedding     ← 이 파일 (RAG 특화 API)

API 키 발급:
    https://dashboard.cohere.com/api-keys (무료)

설치 필요 패키지:
    pip install langchain-cohere
  
플랫폼별 임베딩 모델 종류 
참고: https://docs.langchain.com/oss/python/integrations/embeddings#top-integrations
    
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
import logging
from typing import Dict

from langchain_cohere import CohereEmbeddings

from src.embedding.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class CohereEmbedding(BaseEmbedding):
    """
    Cohere 임베딩 모델 (embed-v4.0).
    
    * 주요 기능:
       - Cohere API 키 검증
       - RAG 특화 다국어 임베딩 (한국어 강함)
       - MRL 기술로 출력 차원 선택 가능 (256/512/1024/1536)
       - 무료 Trial 티어 지원
    
    * 사용 가능 모델 예시:
       - embed-v4.0                    (최신, 다국어+멀티모달)
       - embed-multilingual-v3.0       (다국어, 1024차원)
       - embed-english-v3.0            (영어 전용, 1024차원)
    
    ⚠️ 주의: input_type 필수!
       Cohere는 문서 저장용/질문 검색용 임베딩 구분.
       - 문서 임베딩: "search_document"
       - 질문 임베딩: "search_query"
       LangChain CohereEmbeddings 클래스는 embed_documents/embed_query에서
       자동으로 적절한 input_type을 넣어주므로 신경 안 써도 된다.
    """
    
    # Embed v4.0 지원하는 출력 차원 (이외 값은 에러)
    _ALLOWED_DIMENSIONS = {256, 512, 1024, 1536}
    _DEFAULT_DIMENSION = 1536  # 기본 차원
    
    def __init__(self):
        """초기화 시 모델은 None으로 시작."""
        self.model = None
        self.model_name = None
        self._dimension = None  # 차원 캐싱
    
    def load(self, config: Dict) -> None:
        """
        Cohere 임베딩 모델 초기화.
        
        Args:
            config: config["embedding"]에서 다음 키 사용:
                - embed_name (str): 모델 이름 (예: 'embed-v4.0')
                - embed_dimension (int, optional): 출력 차원 (256/512/1024/1536)
        
        Raises:
            ValueError: COHERE_API_KEY 환경변수 없을 때, 또는 잘못된 차원
        """
        # 1) API 키 검증
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ COHERE_API_KEY 설정되어 있지 않습니다. "
                "https://dashboard.cohere.com/api-keys 에서 무료 발급 후 "
                ".env 파일 추가 및 환경변수 설정 필수."
            )
        
        # 2) 모델 이름 결정
        self.model_name = config["embedding"]["embed_name"]
        
        # TODO: 출력 차원 설정 옵션 필요 시 구현 예정(2026.06.16 minjae)
        # 3) 출력 차원 설정 (MRL, 없으면 기본 1536)
        # dimension = config["embedding"].get("embed_dimension", self._DEFAULT_DIMENSION)
        self._dimension = self._DEFAULT_DIMENSION
        
        # 3-1) 차원 유효성 검증 (Embed v4.0은 4개 값만 허용)
        # if dimension not in self._ALLOWED_DIMENSIONS:
        if self._dimension not in self._ALLOWED_DIMENSIONS:
            raise ValueError(
                # f"❌ Cohere embed-v4.0이 지원하지 않는 차원: {dimension}. "
                f"❌ Cohere embed-v4.0이 지원하지 않는 차원: {self._dimension}. "
                f"지원 차원: {sorted(self._ALLOWED_DIMENSIONS)}"
            )
        # self._dimension = dimension
        
        # 4) LangChain CohereEmbeddings 객체 생성
        self.model = CohereEmbeddings(
            model=self.model_name,
            cohere_api_key=api_key,
        )
        logger.info(
            f"✅ Cohere 임베딩 로딩 완료: {self.model_name} ({self._dimension}차원)"
        )
    
    def get_model(self) -> CohereEmbeddings:
        """초기화된 LangChain CohereEmbeddings 객체 반환."""
        if self.model is None:
            raise RuntimeError("❌ CohereEmbedding.get_model(): load() 먼저 호출해주세요.")
        return self.model
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터 차원 동적 계산 + 캐싱 (한 번만 계산, 이후 캐싱).
        
        * 캐싱 이유: embed_query는 실제 API/모델 호출이라 비용 발생.
                    한 번 계산 후 인스턴스에 저장해서 재사용.
                    
        - [보류] config에서 embed_dimension 지정 시 그 값 사용
        """
        if self._dimension is None:
            self._dimension = len(self.model.embed_query("hello world"))
        return self._dimension