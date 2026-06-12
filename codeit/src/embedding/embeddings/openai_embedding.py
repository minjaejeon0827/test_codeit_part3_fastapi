"""
openai_embedding.py - OpenAI 임베딩

BaseEmbedding 클래스 상속받아 OpenAI 임베딩 모델 사용.(text-embedding-3-large 등.)
전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding               (Strategy 인터페이스)
        ├── HFEmbedding         (로컬 GPU)
        ├── OpenAIEmbedding     ← 이 파일 (유료 API)
        ├── ClaudeEmbedding     (필요 시 구현 예정!)
        ├── GeminiEmbedding     (필요 시 구현 예정!)
        └── GroqEmbedding       (필요 시 구현 예정!)

설치 필요 패키지:
    pip install openai
  
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

from langchain_openai import OpenAIEmbeddings

from src.embedding.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI 임베딩 모델 (기본: text-embedding-3-large).
    
    * 주요 기능:
       - OpenAI API 키 검증
       - text-embedding-3-large 등 임베딩 모델 사용
       - 차원: 3072 (text-embedding-3-large 기준)
    """
    
    # 차원 매핑표 (OpenAI 공식 문서 기준)
    _DIMENSION_MAP = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self):
        """초기화 시 모델은 None으로 시작."""
        self.model = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        OpenAI 임베딩 모델 초기화.
        
        Args:
            config: config["embedding"]에서 model_name 사용
        
        Raises:
            ValueError: OPENAI_API_KEY 환경변수 없을 때
        """
        # 1) API 키 검증
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY 설정되어 있지 않습니다. "
                ".env 파일 확인 및 환경변수 설정 필수."
            )
        
        # 2) 모델 이름 결정 (기본: text-embedding-3-large)
        self.model_name = config["embedding"]["embed_name"]
        
        # embed_model_name = config["embedding"]["embed_name"]
        # OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        
        # 3) LangChain OpenAIEmbeddings 객체 생성
        self.model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=api_key,
        )
        logger.info(f"✅ OpenAI 임베딩 로딩 완료: {self.model_name}")
    
    def get_model(self) -> OpenAIEmbeddings:
        """초기화 된 LangChain OpenAIEmbeddings 객체 반환."""
        if self.model is None:
            raise RuntimeError("❌ load() 먼저 호출해주세요.")
        return self.model
    
    def get_dimension(self) -> int:
        """모델별 차원 반환 (매핑표 사용)."""
        if self.model_name in self._DIMENSION_MAP:
            return self._DIMENSION_MAP[self.model_name]
        
        # 매핑표에 없으면 동적 계산 (Fallback)
        logger.warning(
            f"⚠️ 모델 '{self.model_name}'의 차원 정보 없음 → 동적 계산"
        )
        return len(self.model.embed_query("hello world"))