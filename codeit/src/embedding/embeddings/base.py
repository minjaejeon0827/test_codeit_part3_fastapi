"""
base.py - 모든 임베딩 모델 공통 인터페이스 (추상 클래스)

전략 패턴(Strategy Pattern) 'Strategy' 역할:
- 모든 생성기가 따라야 할 표준 규칙(공통 인터페이스) 정의
- HFEmbedding, OpenAIEmbedding, ClaudeEmbedding, GeminiEmbedding, GroqEmbedding 등은 해당 클래스 상속받아 구현

전략 패턴(Strategy Pattern)에서의 위치:
    BaseEmbedding      ← 이 파일 (Strategy 인터페이스)
        ├── HFEmbedding         (로컬 GPU)
        ├── OpenAIEmbedding     (유료 API)
        ├── OllamaEmbedding     (로컬, 무료)
        └── GeminiEmbedding     (무료 티어 API)

플랫폼별 임베딩 모델 종류
참고: https://docs.langchain.com/oss/python/integrations/embeddings#top-integrations

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEmbedding(ABC):
    """
    모든 임베딩 모델 추상 베이스 클래스.
    
    * 사용 방법:
       1) 해당 클래스 상속받음
       2) load(), get_model(), get_dimension() 메서드 구현 필수!
    
    * 전략 패턴 의미:
       - "텍스트를 벡터로 만든다"는 행위는 같지만,
       - "어떻게(OpenAI? HF?) 만드느냐"는 자식 클래스가 결정
    """
    
    @abstractmethod
    def load(self, config: Dict) -> None:
        """
        임베딩 모델 초기화 (자식 클래스가 반드시 구현).
        
        Args:
            config: 설정 딕셔너리 (embed_type 등 포함)
        """
        pass
    
    @abstractmethod
    def get_model(self) -> Any:
        """실제 LangChain 임베딩 객체 반환 (자식이 구현)."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """임베딩 벡터 차원 수 반환 (FAISS 초기화 시 필요)."""
        pass