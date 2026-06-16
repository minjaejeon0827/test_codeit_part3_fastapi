"""
gemini_embedding.py - Google Gemini 임베딩

BaseEmbedding 클래스 상속받아 Google Generative AI 임베딩 모델 사용.
(models/embedding-001, text-embedding-004 등)
전략 패턴 (Strategy Pattern) 기반 클래스.

⚠️ 모델 마이그레이션 (2026년 6월 기준):
    - embedding-001:      2025년 8월 14일 종료 (사용 불가)
    - text-embedding-004: 2026년 1월 14일 종료 (사용 불가)
    - gemini-embedding-001: ✅ 현재 사용 가능 (유일한 최신 모델)

* gemini-embedding-001 특징:
    - 100개 이상 언어 지원, 입력 토큰 최대 2,048
    - 기본 3072차원 (MRL 기술로 768/1536 등으로 축소 가능)
    - 무료 티어 제공, 유료는 백만 토큰당 $0.15

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding               (Strategy 인터페이스)
        ├── HFEmbedding         (로컬 GPU)
        ├── OpenAIEmbedding     (유료 API)
        ├── OllamaEmbedding     (로컬, 무료)
        ├── GeminiEmbedding     ← 이 파일 (무료 티어 API)
        └── CohereEmbedding     (RAG 특화 API)

API 키 발급:
    https://aistudio.google.com/apikey (무료)

설치 필요 패키지:
    pip install langchain-google-genai
    
embedding-001, text-embedding-004 임베딩 모델 서비스 종료
참고: https://developers.googleblog.com/ko/gemini-embedding-available-gemini-api/

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

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.embedding.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class GeminiEmbedding(BaseEmbedding):
    """
    Google Gemini 임베딩 모델 (gemini-embedding-001).
    
    * 주요 기능:
       - Google API 키 검증
       - 고품질 다국어 임베딩 (한국어 강함, 100개+ 언어)
       - 무료 티어 지원
       - MRL 기술로 출력 차원 축소 가능 (3072 → 768/1536 등)
    
    * 사용 가능 모델 (2026년 6월 기준):
       - gemini-embedding-001   (기본 3072차원, 유일한 최신 모델)
    
    ⚠️ 종료된 모델 (사용 불가):
       - embedding-001          (2025년 8월 종료)
       - text-embedding-004     (2026년 1월 종료)
    """
    
    # 모델별 기본 차원 (gemini-embedding-001 기준)
    _DEFAULT_DIMENSION = 3072
    
    def __init__(self):
        """초기화 시 모델은 None으로 시작."""
        self.model = None
        self.model_name = None
        self._dimension = None  # 차원 캐싱
    
    def load(self, config: Dict) -> None:
        """
        Gemini 임베딩 모델 초기화.
        
        Args:
            config: config["embedding"]에서 다음 키 사용:
                - embed_name (str): 모델 이름 (예: 'gemini-embedding-001')
                - embed_dimension (int, optional): 출력 차원 (MRL 축소용)
        
        Raises:
            ValueError: GOOGLE_API_KEY 환경변수 없을 때, 또는 잘못된 차원
        """
        # 1) API 키 검증
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY 설정되어 있지 않습니다. "
                "https://aistudio.google.com/apikey 에서 무료 발급 후 "
                ".env 파일 추가 및 환경변수 설정 필수."
            )
        
        # 2) 모델 이름 결정
        self.model_name = config["embedding"]["embed_name"]
        
        # 2-1) 종료된 모델 사용 시 경고
        deprecated = {"embedding-001", "text-embedding-004", "models/embedding-001",
                      "models/text-embedding-004", "gemini-embedding-exp-03-07"}
        if self.model_name in deprecated:
            logger.warning(
                f"⚠️ '{self.model_name}'은 서비스 종료된 모델입니다. "
                f"'gemini-embedding-001' 사용을 권장합니다."
            )
        
        # TODO: MRL 차원 축소 옵션 필요 시 구현 예정(2026.06.15 minjae)
        # 3) 출력 차원 설정 (MRL 차원 축소 옵션, 없으면 기본 3072)
        # self._dimension = config["embedding"].get("embed_dimension")
        self._dimension = self._DEFAULT_DIMENSION
        
        # 4) LangChain GoogleGenerativeAIEmbeddings 객체 생성
        self.model = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=api_key,
            output_dimensionality= self._dimension
        )
        logger.info(f"✅ Gemini 임베딩 로딩 완료: {self.model_name}")
    
    def get_model(self) -> GoogleGenerativeAIEmbeddings:
        """초기화된 LangChain GoogleGenerativeAIEmbeddings 객체 반환."""
        if self.model is None:
            raise RuntimeError("❌ GeminiEmbedding.get_model(): load() 먼저 호출해주세요.")
        return self.model
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터 차원 동적 계산 + 캐싱 (한 번만 계산, 이후 캐싱).
        
        * 캐싱 이유: embed_query는 실제 API/모델 호출이라 비용 발생.
                    한 번 계산 후 인스턴스에 저장해서 재사용.
        """
        if self._dimension is None:
            self._dimension = len(self.model.embed_query("hello world"))
        return self._dimension