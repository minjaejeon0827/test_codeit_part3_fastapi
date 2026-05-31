"""
pipeline.py - 임베딩 파이프라인 실행 모듈 

설치 필요 패키지:
    pip install -U langsmith
    pip install langchain-chroma
    pip install langchain-core

    pip install langchain-openai
    pip install langchain-community
    pip install langchain-huggingface
"""


import os
import shutil

from typing import List, Union
from langsmith import traceable
from langchain_chroma import Chroma
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.settings import PROJECT_ROOT
from src.embedding.vector_store import create_vector_store, load_vector_store


def generate_index_name(config: dict) -> str:
    """
    config 설정값 조합하여 벡터DB 고유 인덱스 이름 생성.
    
    조합 요소: 파일 타입 / 분할 방식 / 임베딩 모델 / DB 타입
    예시: "pdf_section_KoE5_faiss" 또는 "all_100_recursive_openai_chroma"

    Args:
        config (dict): 프로젝트 설정 딕셔너리

    Returns:
        str: 파일/폴더 이름으로 안전하게 쓸 수 있는 인덱스 이름 문자열 (예: all_100_recursive_openai_faiss_index)
    """
    data_type = config.get("data", {}).get("file_type", "all")
    splitter = config.get("data", {}).get("splitter", "recursive")
    model = config.get("embedding", {}).get("embed_model", "default")
    db_type = config.get("embedding", {}).get("db_type", "faiss")

    model_key = model.split("/")[-1] if "/" in model else model
    model_key = model_key.replace('-', '_').replace(' ', '_')

    if config.get('data', {}).get('top_k') == 100:
        return f"{data_type}_{config['data']['top_k']}_{splitter}_{model_key}_{db_type}"
    else:
        return f"{data_type}_{splitter}_{model_key}_{db_type}"


@traceable(name="embedding_pipeline")  # LangSmith 추적 이름 명확하게
def run(
    config: dict,
    chunks: List[Document],
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
    is_save: bool = False,
    session_id: str = None
) -> Union[FAISS, Chroma]:
    """
    임베딩 파이프라인 실행: 청크 → 임베딩 → 벡터 저장소 생성 또는 로딩
    
    is_save=True인 경우 새 벡터DB 생성하고, False인 경우 기존 벡터DB 로드.

    Args:
        config (dict): 설정 정보
        chunks (List[Document]): Document 객체 리스트
        embeddings (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): 초기화된 임베딩 객체
        is_save (bool): 저장 모드 여부

    Returns:
        Union[FAISS, Chroma]: 생성되거나 로드된 벡터 저장소 인스턴스

    Raises:
        ValueError: 잘못된 인자나 지원하지 않는 DB 타입일 경우
    """

    if not isinstance(chunks, list) or not all(isinstance(chunk, Document) for chunk in chunks):
        raise ValueError("❌ (embedding.embedding_main.chunks) chunks는 Document 객체의 리스트여야 합니다.")
    if len(chunks) == 0:
        raise ValueError("❌ (embedding.embedding_main.chunks) chunks 리스트가 비어 있음")

    if embeddings is None or not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ (embedding.embedding_main.embeddings) 잘못된 embeddings 인자")

    embed_config = config['embedding']
    db_type = embed_config['db_type'].lower()
    vector_store_path = os.path.join(PROJECT_ROOT, embed_config.get("vector_store_path", "data"))

    if not isinstance(vector_store_path, str) or vector_store_path.strip() == "":
        raise ValueError("❌ (embedding.embedding_main.vector_store_path) 잘못된 vector_store_path 경로")

    os.makedirs(vector_store_path, exist_ok=True)
    index_name = generate_index_name(config)
    index_name = index_name + f"_{session_id}"

    if not isinstance(index_name, str) or index_name.strip() == "":
        raise ValueError("❌ (embedding.embedding_main.index_name) 잘못된 index_name 생성")

    if db_type == "faiss":
        faiss_file = os.path.join(vector_store_path, f"{index_name}.faiss")
        pkl_file = os.path.join(vector_store_path, f"{index_name}.pkl")
        db_exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)

    elif db_type == "chroma":
        chroma_dir = os.path.join(vector_store_path, index_name)
        sqlite_path = os.path.join(chroma_dir, "chroma.sqlite3")

        has_sqlite = os.path.exists(sqlite_path)
        has_index_dirs = any(
            os.path.isdir(os.path.join(chroma_dir, d)) and len(os.listdir(os.path.join(chroma_dir, d))) >= 4
            for d in os.listdir(chroma_dir)
            if os.path.isdir(os.path.join(chroma_dir, d))
        ) if os.path.exists(chroma_dir) else False

        db_exists = has_sqlite and has_index_dirs

        if os.path.exists(chroma_dir) and not db_exists:
            print("⚠️ 불완전한 Chroma 벡터 DB가 감지되어 삭제합니다.")
            shutil.rmtree(chroma_dir)
            db_exists = False

    else:
        raise ValueError(f"❌ (embedding.embedding_main.db_type) 지원하지 않는 DB 타입입니다: {db_type}")

    if is_save:
        vector_store = create_vector_store(chunks, embeddings, index_name, db_type, output_path=vector_store_path)
        print("✅ Vector DB 생성 완료")
    else:
        vector_store = load_vector_store(vector_store_path, embeddings, index_name, db_type)
        print("✅ Vector DB 로드 완료")

    return vector_store