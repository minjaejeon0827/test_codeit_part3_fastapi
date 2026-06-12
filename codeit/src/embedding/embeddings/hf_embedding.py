"""
hf_embedding.py - HuggingFace 임베딩

BaseEmbedding 클래스 상속받아 HuggingFace 임베딩 모델 사용. (KoE5, BGE, E5 등)
전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding               (Strategy 인터페이스)
        ├── HFEmbedding         ← 이 파일 (로컬 GPU)
        ├── OpenAIEmbedding     (유료 API)
        ├── ClaudeEmbedding     (필요 시 구현 예정!)
        ├── GeminiEmbedding     (필요 시 구현 예정!)
        └── GroqEmbedding       (필요 시 구현 예정!)

플랫폼별 임베딩 모델 종류 
참고: https://docs.langchain.com/oss/python/integrations/embeddings#top-integrations

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import logging
from typing import Dict

from langchain_huggingface import HuggingFaceEmbeddings

# 내부 임포트
from src.embedding.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class HFEmbedding(BaseEmbedding):
    """
    HuggingFace 임베딩 모델 (KoE5, BGE, E5 등).
    
    * 주요 기능:
       - HuggingFace Hub 모델 로딩
       - 한국어 모델 (KoE5) 등 다양한 모델 지원
       - 차원은 모델마다 다름 (동적 계산)
    """
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self._dimension = None  # 차원 캐싱
    
    def load(self, config: Dict) -> None:
        """
        HuggingFace 임베딩 모델 로딩.
        
        Args:
            config: config["embedding"]["embed_model"] 사용 (예: 'nlpai-lab/KoE5')
        """
        
        self.model_name = config["embedding"]["embed_name"]
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)
        logger.info(f"✅ HuggingFace 임베딩 로딩 완료: {self.model_name}")
    
    def get_model(self) -> HuggingFaceEmbeddings:
        """초기화 된 LangChain HuggingFaceEmbeddings 객체 반환."""
        if self.model is None:
            raise RuntimeError("❌ HFEmbedding.get_model(): load() 먼저 호출해주세요.")
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