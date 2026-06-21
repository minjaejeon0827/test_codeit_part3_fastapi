"""
ollama_embedding.py - Ollama 로컬 임베딩

BaseEmbedding 클래스 상속받아 Ollama 로컬 임베딩 모델 사용.
(nomic-embed-text, mxbai-embed-large 등)
전략 패턴 (Strategy Pattern) 기반 클래스.

* Ollama 강점:
    - 완전 무료 (로컬 실행)
    - API 키 불필요
    - 데이터 외부로 안 나감 (보안 좋음)
    - 다양한 오픈소스 임베딩 모델 지원

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding               (Strategy 인터페이스)
        ├── HFEmbedding         (로컬 GPU)
        ├── OpenAIEmbedding     (유료 API)
        ├── OllamaEmbedding     ← 이 파일 (로컬, 무료)
        ├── GeminiEmbedding     (무료 티어 API)
        └── CohereEmbedding     (RAG 특화 API)

Ollama 사전 준비:
    1) Ollama 설치 (https://ollama.com/download)
    2) 임베딩 모델 받기
    ollama pull nomic-embed-text  -> 추천 (가볍고 빠름)
    ollama pull mxbai-embed-large -> 더 고품질

    3) Ollama 서버 실행 (보통 자동 실행)
    ollama serve

설치 필요 패키지:
    pip install langchain-ollama
  
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

from langchain_ollama import OllamaEmbeddings

from src.embedding.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class OllamaEmbedding(BaseEmbedding):
    """
    Ollama 로컬 임베딩 모델 (nomic-embed-text 등).
    
    * 주요 기능:
       - 로컬에서 실행되는 무료 임베딩
       - API 키 불 필요
       - 차원은 모델마다 다름 (동적 계산)
    
    * 지원 모델 예시:
       - nomic-embed-text   (768차원, 가볍고 빠름)
       - mxbai-embed-large  (1024차원, 고품질)
       - bge-m3             (1024차원, 다국어)
    """
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.base_url = None
        self._dimension = None  # 차원 캐싱
    
    def load(self, config: Dict) -> None:
        """
        Ollama 임베딩 모델 로딩.
        
        Args:
            config: config["embedding"]["embed_name"] 사용 (예: 'nomic-embed-text')
        
        Raises:
            RuntimeError: Ollama 서버 연결 실패 시
        """
        self.model_name = config["embedding"]["embed_name"]

        # 변경: config 아니라 환경변수(.env)에서 읽기
        #      - 환경마다 서버 주소가 다름 (localhost / 원격 GPU / 도커)
        #      - 없으면 기본값 localhost 사용        
        # base_url 선택 (기본: http://localhost:11434)
        # base_url = config["embedding"].get("ollama_base_url", "http://localhost:11434")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        self.model = OllamaEmbeddings(
            model=self.model_name,
            base_url=self.base_url,
        )
        logger.info(f"✅ Ollama 임베딩 로딩 완료: {self.model_name}")
    
    def get_model(self) -> OllamaEmbeddings:
        """초기화된 LangChain OllamaEmbeddings 객체 반환."""
        if self.model is None:
            raise RuntimeError("❌ OllamaEmbedding.get_model(): load() 먼저 호출해주세요.")
        return self.model
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터 차원 동적 계산 + 캐싱. (한 번만 계산, 이후 캐싱).
        
        * 캐싱 이유: embed_query는 실제 모델 호출이라 비용 발생.
                    한 번 계산 후 인스턴스에 저장해서 재사용.
        """
        if self._dimension is None:
            self._dimension = len(self.model.embed_query("hello world"))
        return self._dimension