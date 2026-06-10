"""
faiss_store.py - FAISS 벡터 저장소

BaseVectorStore 클래스 상속받아 FAISS 벡터 저장소 사용.
전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴(Strategy Pattern)에서의 위치:
    BaseVectorStore        (Strategy 인터페이스)
        ├── FAISSStore     ← 이 파일
        ├── ChromaStore
        ├── PineconeStore  (필요 시 구현 예정!)
        ├── QdrantStore    (필요 시 구현 예정!)
        └── WeaviateStore  (필요 시 구현 예정!)
        
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
import logging
from typing import Any, List

import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# 내부 임포트
from src.embedding.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)

class FAISSStore(BaseVectorStore):
    """
    FAISS 기반 벡터 저장소.
    
    🎯 특징:
        - 로컬 파일 저장 ({index_name}.faiss + {index_name}.pkl)
        - 빠른 유사도 검색
        - 메모리 효율적
    """
    
    def create(
        self,
        chunks: List[Document],
        embedding_model: Any,
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], 
        index_name: str,
        output_path: str,
    ) -> FAISS:
        """FAISS 인덱스 생성 및 디스크 저장."""
        # 1) 임베딩 차원 자동 계산
        dimension = len(embedding_model.embed_query("hello world"))
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # dimension = 3072 if isinstance(embeddings, OpenAIEmbeddings) else len(embeddings.embed_query("hello world"))
        logger.info(f"📌 FAISS 인덱스 차원: {dimension}")
        
        # 2) 빈 FAISS 인덱스 생성
        vector_store = FAISS(
            embedding_function=embedding_model,
            # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
            # embedding_function=embeddings,
            index=faiss.IndexFlatL2(dimension),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # 3) 배치로 문서 추가 (부모 클래스 공통 헬퍼 함수 호출!)
        vector_store = self._add_docs_in_batch(vector_store, chunks)
        
        # 4) 디스크에 저장
        os.makedirs(output_path, exist_ok=True)
        vector_store.save_local(
            folder_path=output_path, 
            index_name=index_name,
        )
        logger.info(f"✅ FAISS 벡터 DB 저장 완료: {index_name}")
        
        return vector_store
    
    def load(
        self,
        path: str,
        embedding_model: Any,
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
        index_name: str,
    ) -> FAISS:
        """저장 된 FAISS 인덱스 로드."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ [FileNotFound] 존재하지 않는 벡터 DB 경로: {path}")
        
        return FAISS.load_local(
            folder_path=path,
            index_name=index_name,
            embeddings=embedding_model,
            # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
            # embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    
    # TODO: main_page.py -> VectorStorePaths @dataclass -> get_vector_store_paths 함수와 비슷한 기능이므로 추후 확인 필요(2026.06.10 minjae)
    def exists(self, path: str, index_name: str) -> bool:
        """FAISS 파일(.faiss + .pkl) 존재 여부 확인."""
        faiss_file = os.path.join(path, f"{index_name}.faiss")
        pkl_file = os.path.join(path, f"{index_name}.pkl")
        return os.path.exists(faiss_file) and os.path.exists(pkl_file)