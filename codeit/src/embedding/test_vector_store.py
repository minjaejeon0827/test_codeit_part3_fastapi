"""

설치 필요 패키지:
    pip install -U langsmith
    pip install langchain-chroma
    pip install langchain-core
    pip install faiss-cpu

    pip install langchain-openai
    pip install langchain-community
    pip install langchain-huggingface
"""

import os
import faiss
import shutil

from tqdm import tqdm
from typing import List, Union, Optional

from langsmith import traceable
from langchain_chroma import Chroma
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores.base import VectorStore
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore


def create_embedding_model(embed_model_name: str) -> Union[OpenAIEmbeddings, HuggingFaceEmbeddings]:
    """
    주어진 모델 이름에 따라 임베딩 모델을 초기화합니다.

    Args:
        embed_model_name (str): 사용할 임베딩 모델 이름 ('openai' 또는 Hugging Face 모델 이름)

    Returns:
        Union[OpenAIEmbeddings, HuggingFaceEmbeddings]: 초기화된 임베딩 모델 객체

    Raises:
        ValueError: API 키 누락 또는 모델 초기화 실패 시 발생
    """
    try:
        if embed_model_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ [Value] (vector_db.create_embedding_model.api_key) OPENAI_API_KEY 누락")
            return OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        else:
            return HuggingFaceEmbeddings(model_name=embed_model_name)
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.create_embedding_model.init) 임베딩 모델 초기화 실패 원인: {e}")


@traceable(name="create_vector_store")
def create_vector_store(
    all_chunks: List[Document],
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
    index_name: str,
    db_type: str = "faiss",
    is_save: bool = True,
    output_path: str = ""
) -> Union[FAISS, Chroma]:
    """
    문서 청크 리스트를 이용해 FAISS 또는 Chroma 기반 벡터 DB를 생성하고 저장합니다.

    Args:
        all_chunks (List[Document]): 임베딩할 문서 청크 리스트
        embeddings (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): 초기화된 임베딩 객체
        index_name (str): 저장할 벡터 DB 인덱스 이름
        db_type (str): 벡터 DB 유형 ('faiss' 또는 'chroma')
        is_save (bool): DB 저장 여부 (faiss만 해당)
        output_path (str): 벡터 DB 저장 경로 (기본값: 프로젝트 루트/data/vector_db)

    Returns:
        Union[FAISS, Chroma]: 생성된 벡터 DB 인스턴스

    Raises:
        ValueError: 잘못된 입력 값 또는 지원하지 않는 DB 타입
        RuntimeError: 벡터 DB 생성 중 오류 발생 시
    """
    if not all_chunks or not isinstance(all_chunks, list):
        raise ValueError("❌ [Value] (vector_db.create_vector_store.all_chunks) 비어 있거나 잘못된 Document 리스트")

    if not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ [Value] (vector_db.create_vector_store.embeddings) 잘못된 임베딩 객체")

    if not index_name:
        raise ValueError("❌ [Value] (vector_db.create_vector_store.index_name) 빈 index_name 인자")

    if not output_path:
        raise ValueError("❌ [Value] (vector_db.create_vector_store.output_path) 빈 output_path 인자")

    db_type = db_type.lower()

    print(f"📌 [Info] (vector_db.create_vector_store) 임베딩 모델: {embeddings.__class__.__name__}")

    try:
        dimension = 3072 if isinstance(embeddings, OpenAIEmbeddings) else len(embeddings.embed_query("hello world"))
    except Exception as e:
        raise ValueError(f"❌ [Value] (vector_db.create_vector_store.dimension) 임베딩 차원 계산 실패 원인: {e}")

    try:
        os.makedirs(output_path, exist_ok=True)

        if db_type == "faiss":
            print(f"📌 [Info] (vector_db.create_vector_store) 벡터 DB 유형: {db_type}")
            vector_store = FAISS(
                embedding_function=embeddings,
                index=faiss.IndexFlatL2(dimension),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            if is_save:
                vector_store.save_local(folder_path=output_path, index_name=index_name)
                print("✅ [Success] (vector_db.create_vector_store) FAISS 벡터 DB 저장 완료")

        elif db_type == "chroma":
            print(f"📌 [Info] (vector_db.create_vector_store) 벡터 DB 유형: {db_type}")
            chroma_path = os.path.join(output_path, index_name)
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
                print(f"⚠️ [Warning] (vector_db.create_vector_store) 기존 Chroma DB 제거 완료")

            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
            vector_store = add_docs_in_batch(vector_store, all_chunks)
            print("✅ [Success] (vector_db.create_vector_store) Chroma 벡터 DB 저장 완료")

        else:
            raise ValueError("❌ [Value] (vector_db.create_vector_store.db_type) 지원하지 않는 벡터 DB 타입 ('faiss' 또는 'chroma'만 가능)")

        return vector_store

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.create_vector_store.general) 벡터 DB 생성 실패 원인: {e}")


@traceable(name="load_vector_store")
def load_vector_store(
    path: str,
    embeddings: Union[HuggingFaceEmbeddings, OpenAIEmbeddings],
    index_name: str,
    db_type: str = "faiss"
) -> Union[FAISS, Chroma]:
    """
    저장된 FAISS 또는 Chroma 벡터 DB를 로컬에서 로드합니다.

    Args:
        path (str): 벡터 DB 루트 디렉토리 경로
        embeddings (Union[HuggingFaceEmbeddings, OpenAIEmbeddings]): 초기화된 임베딩 객체
        index_name (str): 불러올 벡터 DB 인덱스 이름
        db_type (str): 벡터 DB 유형 ('faiss' 또는 'chroma')

    Returns:
        Union[FAISS, Chroma]: 로드된 벡터 DB 인스턴스

    Raises:
        FileNotFoundError: 지정 경로가 존재하지 않는 경우
        ValueError: 지원하지 않는 DB 타입인 경우
        RuntimeError: 로딩 중 오류 발생 시
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ [FileNotFound] (vector_db.load_vector_store.path) 존재하지 않는 벡터 DB 경로: {path}")

    if not isinstance(embeddings, (HuggingFaceEmbeddings, OpenAIEmbeddings)):
        raise ValueError("❌ [Value] (vector_db.load_vector_store.embeddings) 잘못된 임베딩 객체")

    if not index_name:
        raise ValueError("❌ [Value] (vector_db.load_vector_store.index_name) 빈 index_name 인자")

    db_type = db_type.lower()

    try:
        if db_type == "faiss":
            return FAISS.load_local(
                folder_path=path,
                index_name=index_name,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        elif db_type == "chroma":
            chroma_path = os.path.join(path, index_name)
            return Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_path,
                collection_name="chroma_db",
            )
        else:
            raise ValueError(f"❌ [Value] (vector_db.load_vector_store.db_type) 지원하지 않는 벡터 DB 타입: {db_type}")

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.load_vector_store.general) 벡터 DB 로드 실패 원인: {e}")


def add_docs_in_batch(
    vector_store: VectorStore,
    chunks: Optional[List[Document]],
    batch_size: int = 128
) -> VectorStore:
    """
    문서 청크 리스트를 지정된 배치 크기로 나누어 벡터 스토어에 추가합니다.

    Args:
        vector_store (VectorStore): 문서가 삽입될 벡터 저장소 객체
        chunks (Optional[List[Document]]): 삽입할 문서 청크 리스트
        batch_size (int): 한 번에 처리할 문서 수

    Returns:
        VectorStore: 문서가 삽입된 벡터 저장소 인스턴스

    Raises:
        ValueError: chunks가 None이거나 잘못된 경우, batch_size가 유효하지 않은 경우
        RuntimeError: 문서 삽입 중 오류 발생 시
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("❌ [Value] (vector_db.add_docs_in_batch.chunks) 비어 있거나 잘못된 Document 리스트")

    if batch_size <= 0:
        raise ValueError("❌ [Value] (vector_db.add_docs_in_batch.batch_size) batch_size는 1 이상이어야 함")

    total = len(chunks)
    pbar = tqdm(
        range(0, total, batch_size),
        desc=f"📌 [Info] (vector_db.add_docs_in_batch) {vector_store.__class__.__name__} 인덱싱 진행 중",
        unit="batch",
    )

    try:
        for i in pbar:
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)

            end_idx = min(i + batch_size, total)
            pbar.set_postfix_str(f"진행 {end_idx} / {total}")

        return vector_store

    except Exception as e:
        raise RuntimeError(f"❌ [Runtime] (vector_db.add_docs_in_batch.general) 문서 배치 삽입 실패 원인: {e}")