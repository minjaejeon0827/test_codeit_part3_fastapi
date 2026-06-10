"""
pipeline.py - 임베딩 파이프라인 (전략 + 팩토리 패턴 적용 버전)

변경 사항:
    - 모든 if/else 분기 제거 (전략 패턴 알아서 처리)
    - 새 임베딩/Vector DB 추가 매우 쉬워짐
    - 책임 명확히 분리:
        * embedding/  → 임베딩 모델 관리
        * stores/     → 벡터 저장소 관리
        * pipeline.py → 둘 조합해 파이프라인 실행

호출 예시:
    from src.embedding.embeddings.factory import create_embedding
    from src.embedding.pipeline import run
    
    embedding = create_embedding(config)
    vector_store = run(config, chunks, embedding, is_save=True, session_id=sid)

설치 필요 패키지:
    pip install -U langsmith
    pip install langchain-chroma
    pip install langchain-core

    pip install langchain-openai
    pip install langchain-community
    pip install langchain-huggingface
    
파이썬 디자인 패턴 
-> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
-> 생성 패턴 -> 팩토리 메소드 패턴(Factory Method Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
import logging
from typing import List, Union

from langsmith import traceable
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from src.settings import PROJECT_ROOT
from src.embedding.embeddings.base import BaseEmbedding
from src.embedding.stores.factory import create_vector_store

logger = logging.getLogger(__name__)


def generate_index_name(config: dict) -> str:
    """
    config 설정값 조합 및 벡터 DB 인덱스 이름 생성.
    (기존 로직 그대로 유지)
    
    예시: 'all_recursive_KoE5_faiss' 또는 'pdf_100_section_openai_chroma'
    """
    data_type = config.get("data", {}).get("file_type", "all")
    splitter = config.get("data", {}).get("splitter", "recursive")
    model = config.get("embedding", {}).get("embed_model", "default")
    db_type = config.get("embedding", {}).get("db_type", "faiss")
    
    model_key = model.split("/")[-1] if "/" in model else model
    model_key = model_key.replace('-', '_').replace(' ', '_')
    
    if config.get('data', {}).get('top_k') == 100:
        return f"{data_type}_{config['data']['top_k']}_{splitter}_{model_key}_{db_type}"
    return f"{data_type}_{splitter}_{model_key}_{db_type}"


@traceable(name="embedding_pipeline")
def run(
    config: dict,
    chunks: List[Document],
    embedding: BaseEmbedding,
    # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
    # embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], # 글자를 숫자로 바꿀 번역기 (둘 중 하나)
    is_save: bool = False,
    session_id: str = None,
) -> Union[FAISS, Chroma]:
    """
    임베딩 파이프라인 실행: 청크 → 임베딩 → 벡터 DB 저장소 생성/로드.
    
    변경 사항 (vs embedding_main):
        - if/else 분기 모두 사라짐 (전략 패턴이 캡슐화)
        - embedding이 객체로 들어옴 (이전: LangChain 객체 직접)
        - 코드 길이 50% 감소
    
    Args:
        config: 프로젝트 설정 (embedding 섹션 사용)
        chunks: 임베딩할 문서 청크 리스트
        embedding: 미리 로딩된 BaseEmbedding 클래스 객체 (전략 패턴)
        is_save: True면 새로 생성, False면 기존 로드
        session_id: 세션 식별자 (인덱스 이름에 포함)
    
    Returns:
        FAISS 또는 Chroma 벡터 DB 저장소 인스턴스
    
    Raises:
        ValueError: 잘못된 인자
    """
    # 1) 입력 검증
    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        # raise ValueError("❌ chunks Document 리스트여야 합니다.")
        raise ValueError("❌ (embedding.embedding_pipeline.chunks) chunks는 Document 객체 리스트여야 합니다.")
    # if embeddings is None or not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
    if not chunks:
        # raise ValueError("❌ chunks 리스트 비어 있습니다.")
        raise ValueError("❌ (embedding.embedding_pipeline.embeddings) 잘못된 embeddings 인자")
    
    # 2) 경로 + 인덱스 이름 준비
    # TODO: 필요 시 아래 주석친 코드 참고(2026.06.10 minjae)
    # embed_config = config["embedding"]
    
    vector_store_path = os.path.join(
        PROJECT_ROOT,
        # embed_config.get("vector_store_path", "data/vector_store"),
        config["embedding"]["db_type"].get("vector_store_path", "data/vector_store"),
    )
    os.makedirs(vector_store_path, exist_ok=True)
    
    index_name = generate_index_name(config)
    if session_id:
        index_name = f"{index_name}_{session_id}"
    
    logger.info(f"📌 벡터 DB 인덱스 이름: {index_name}")
    
    # 3) 벡터 DB 저장소 객체 생성 (어떤 DB인지 신경 X!)
    # store = create_vector_store(embed_config["db_type"])
    store = create_vector_store(config)
    
    # 4) 임베딩 모델 추출
    embedding_model = embedding.get_model()
    
    # 5) 벡터 DB 생성 또는 로드
    if is_save:
        vector_store = store.create(
            chunks=chunks,
            embedding_model=embedding_model,
            index_name=index_name,
            output_path=vector_store_path,
        )
        logger.info("✅ [Success] Vector DB 생성 완료")
    else:
        vector_store = store.load(
            path=vector_store_path,
            embedding_model=embedding_model,
            index_name=index_name,
        )
        logger.info("✅ [Success] Vector DB 로드 완료")
    
    return vector_store