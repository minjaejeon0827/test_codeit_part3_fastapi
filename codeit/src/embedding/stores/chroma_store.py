"""
chroma_store.py - Chroma 벡터 저장소

BaseVectorStore 클래스 상속받아 Chroma 벡터 저장소 사용.
전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴(Strategy Pattern)에서의 위치:
    BaseVectorStore        (Strategy 인터페이스)
        ├── FAISSStore
        ├── ChromaStore    ← 이 파일
        ├── PineconeStore  (필요 시 구현 예정!)
        ├── QdrantStore    (필요 시 구현 예정!)
        └── WeaviateStore  (필요 시 구현 예정!)
        
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
import shutil
import logging
from typing import Any, List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.embedding.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)

class ChromaStore(BaseVectorStore):
    """
    Chroma 기반 벡터 저장소.
    
    * 특징:
       - SQLite + 인덱스 폴더 저장
       - 메타데이터 필터링 강력
       - 영속성 좋음
    """
    
    COLLECTION_NAME = "chroma_db"  # 클래스 상수 분리 (매직 스트링 방지)
    
    def create(
        self,
        chunks: List[Document],
        embedding_model: Any,
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
        index_name: str,
        output_path: str,
    ) -> Chroma:
        """Chroma DB 생성 (기존 폴더 자동 정리)."""
        chroma_path = os.path.join(output_path, index_name)
        
        # 1) 기존 DB 정리 (있다면)
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            logger.warning(f"⚠️ [Warning] 기존 Chroma DB 제거 완료: {chroma_path}")
        
        # 2) 빈 Chroma DB 생성
        vector_store = Chroma(
            embedding_function=embedding_model,
            # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
            # embedding_function=embeddings,
            persist_directory=chroma_path,
            collection_name=self.COLLECTION_NAME,
        )
        
        # 3) 배치로 문서 추가 (부모 클래스 공통 헬퍼 함수 호출!)
        vector_store = self._add_docs_in_batch(vector_store, chunks)
        logger.info(f"✅ [Success] Chroma 벡터 DB 저장 완료: {index_name}")
        
        return vector_store
    
    def load(
        self,
        path: str,
        embedding_model: Any,
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
        index_name: str,
    ) -> Chroma:
        """저장 된 Chroma DB 로드."""
        # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
        # chroma_path = os.path.join(path, index_name)
        # if not os.path.exists(chroma_path):
            # raise FileNotFoundError(f"❌ [FileNotFound] 존재하지 않는 Chroma DB 경로: {chroma_path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ [FileNotFound] 존재하지 않는 Chroma DB 경로: {chroma_path}")
        
        chroma_path = os.path.join(path, index_name)
        return Chroma(
            embedding_function=embedding_model,
            persist_directory=chroma_path,
            collection_name=self.COLLECTION_NAME,
        )
    
    # TODO: main_page.py -> VectorStorePaths @dataclass -> get_vector_store_paths 함수와 비슷한 기능이므로 추후 확인 필요(2026.06.10 minjae)
    def exists(self, path: str, index_name: str) -> bool:
        """
        Chroma DB 존재 여부 확인.
        
        * 확인 항목:
           1) chroma.sqlite3 파일 존재
           2) 인덱스 폴더 존재 + 내부 파일 4개 이상
        
        * 자동 정리:
           불완전한 DB 감지 시 자동 삭제 (재생성 강제)
        """
        chroma_dir = os.path.join(path, index_name)
        
        # 폴더 없으면 False
        if not os.path.exists(chroma_dir):
            return False
        
        # 1) SQLite 파일 확인
        sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")
        has_sqlite = os.path.exists(sqlite_path)
        
        # 2) 인덱스 폴더 (UUID 폴더) 존재 + 내부 파일 4개 이상
        has_index_dirs = any(
            os.path.isdir(os.path.join(chroma_dir, d)) 
            and len(os.listdir(os.path.join(chroma_dir, d))) >= 4
            for d in os.listdir(chroma_dir)
            if os.path.isdir(os.path.join(chroma_dir, d))
        )
        
        db_exists = has_sqlite and has_index_dirs
        
        # 불완전한 DB 자동 정리
        if not db_exists:
            logger.warning(
                f"⚠️ [Warning] 불완전한 Chroma DB 감지 → 삭제: {chroma_dir}"
            )
            shutil.rmtree(chroma_dir)
        
        return db_exists