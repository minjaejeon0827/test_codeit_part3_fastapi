"""
factory.py - 벡터 DB 저장소 생성 팩토리

팩토리 메소드 패턴(Factory Method Pattern):
    - db_type 문자열만 받으면 적절한 Store 객체 생성 및 반환
    - 호출자는 어떤 클래스가 생성되는지 신경 쓸 필요 없음
    - 새로운 Vector DB 추가 시 _STORES 딕셔너리 한 줄만 추가!

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import Dict

from src.embedding.stores.base import BaseVectorStore
from src.embedding.stores.faiss_store import FAISSStore
from src.embedding.stores.chroma_store import ChromaStore

# "Vector DB 타입 문자열 → 클래스" 매핑표
_STORES: Dict[str, type] = {
    "faiss": FAISSStore,
    "chroma": ChromaStore,
    # TODO: 추후 필요 시 PineconeStore, QdrantStore, WeaviateStore 클래스 구현 예정(2026.06.10 minjae)
    # "pinecone": PineconeStore,
    # "qdrant": QdrantStore,
    # "weaviate": WeaviateStore,
}

# def create_vector_store(db_type: str) -> BaseVectorStore:
def create_vector_store(config: Dict) -> BaseVectorStore:
    """
    config에 따라 적절한 BaseVectorStore 객체 생성 및 반환.
    
    * 사용 예시:
        store = create_vector_store(config)
        if store.exists(path, index_name):
            vs = store.load(path, embedding_model, index_name)
        else:
            vs = store.create(chunks, embedding_model, index_name, path)
    
    Args:
        config: 프로젝트 설정 딕셔너리
        db_type: 'faiss', 'chroma', 'pinecone', 'qdrant', 'weaviate' 등
    
    Returns:
        store_class(): BaseVectorStore 클래스 객체
    
    Raises:
        ValueError: 지원하지 않는 Vector DB 타입인 경우
    """
    # db_type = db_type.lower().strip()
    db_type = config["embedding"]["db_type"].lower().strip()
    
    if db_type not in _STORES:
        supported = list(_STORES.keys())
        raise ValueError(
            f"❌ 지원하지 않는 Vector DB 타입: '{db_type}'. "
            f"지원되는 타입: {supported}"
        )
    
    store_class = _STORES[db_type]
    return store_class()