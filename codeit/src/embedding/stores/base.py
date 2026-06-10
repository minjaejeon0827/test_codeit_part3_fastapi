"""
base.py - 모든 벡터 저장소 공통 인터페이스 (추상 클래스)

전략 패턴(Strategy Pattern) 'Strategy' 역할:
- 모든 생성기가 따라야 할 표준 규칙(공통 인터페이스) 정의
- FAISSStore, ChromaStore, PineconeStore, QdrantStore, WeaviateStore 등은 해당 클래스 상속받아 구현

전략 패턴(Strategy Pattern)에서의 위치:
    BaseVectorStore      ← 이 파일 (Strategy 인터페이스)
        ├── FAISSStore
        ├── ChromaStore
        ├── PineconeStore  (필요 시 구현 예정!)
        ├── QdrantStore    (필요 시 구현 예정!)
        └── WeaviateStore  (필요 시 구현 예정!)
        
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from abc import ABC, abstractmethod
from typing import Any, List

from tqdm import tqdm
from langchain_core.documents import Document

class BaseVectorStore(ABC):
    """
    모든 벡터 저장소 추상 베이스 클래스.
    
    * 전략 패턴 의미:
       - "벡터 저장/로드한다"는 행위는 같지만,
       - "어떻게(FAISS? Chroma?) 하느냐"는 자식 클래스가 결정
    
    * 자식 클래스가 구현해야 하는 3가지:
       - create(): 새로 벡터 DB 생성
       - load():   기존 벡터 DB 로드
       - exists(): 벡터 DB 존재 여부 확인
    """
    
    @abstractmethod
    def create(
        self, 
        chunks: List[Document], 
        embedding_model: Any, 
        index_name: str, 
        output_path: str,
    ) -> Any:
        """
        벡터 저장소 생성 (자식 클래스 구현).
        
        Args:
            chunks: 임베딩할 문서 청크 리스트
            embedding_model: LangChain 임베딩 객체
            index_name: 인덱스 이름 (파일/폴더 이름 사용)
            output_path: 저장 경로
        
        Returns:
            생성된 벡터 저장소 인스턴스
        """
        pass
    
    @abstractmethod
    def load(
        self, 
        path: str, 
        embedding_model: Any, 
        index_name: str,
    ) -> Any:
        """저장 된 벡터 저장소 로드 (자식 클래스 구현)."""
        pass
    
    @abstractmethod
    def exists(self, path: str, index_name: str) -> bool:
        """벡터 저장소 존재 여부 확인 (자식 클래스 구현)."""
        pass
    
    # ============================================================
    # 공통 헬퍼 (모든 자식 클래스 사용 - 코드 중복 방지)
    # ============================================================
    
    def _add_docs_in_batch(
        self,
        vector_store: Any,
        chunks: List[Document],
        batch_size: int = 128,
    ) -> Any:
        """
        배치 단위 문서 추가 (모든 저장소 공통).
        
        * 부모 클래스 구현 사유:
            FAISS, Chroma 등 모두 같은 로직이라 중복 방지.
        
        Args:
            vector_store: 문서 삽입할 벡터 저장소
            chunks: 삽입할 문서 청크 리스트
            batch_size: 한 번에 처리할 문서 수 (기본 128)
        
        Returns:
            vector_store: 문서 삽입된 벡터 저장소
        """
        if not chunks:
            raise ValueError("❌ chunks 비어 있습니다.")
        if batch_size <= 0:
            raise ValueError("❌ batch_size 1 이상이어야 합니다.")
        
        total = len(chunks)
        pbar = tqdm(
            range(0, total, batch_size),
            desc=f"📌 [Info] {vector_store.__class__.__name__} 인덱싱 진행 중",
            unit="batch",
        )
        
        for i in pbar:
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)
            end_idx = min(i + batch_size, total)
            pbar.set_postfix_str(f"진행 {end_idx} / {total}")
        
        return vector_store