"""
factory.py - 임베딩 모델 객체 생성 팩토리

팩토리 메소드 패턴(Factory Method Pattern):
    - model_type 문자열만 받으면 적절한 Embedding 객체 생성 및 반환
    - 호출자는 어떤 클래스가 생성되는지 신경 쓸 필요 없음
    - 새 임베딩 모델 추가 시 _EMBEDDINGS 딕셔너리 한 줄만 추가!

파이썬 디자인 패턴 -> 생성 패턴 -> 팩토리 메소드 패턴(Factory Method Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import Dict

from src.embedding.embeddings.base import BaseEmbedding
from src.embedding.embeddings.openai_embedding import OpenAIEmbedding
from src.embedding.embeddings.hf_embedding import HFEmbedding

# "임베딩 모델 타입 문자열 → 클래스" 매핑표
# 임베딩 모델 새로 추가 시 클래스 import + _EMBEDDINGS 딕셔너리 객체 한 줄 추가!
_EMBEDDINGS: Dict[str, type] = {
    "openai": OpenAIEmbedding,
    "huggingface": HFEmbedding,
    # TODO: 추후 필요시 아래 주석친 코드 사용 예정(2026.06.12 minjae)
    # "claude": ClaudeEmbedding,
    # "gemini": GeminiEmbedding,
    # "groq": GroqEmbedding,
}

def create_embedding(config: Dict) -> BaseEmbedding:
    """
    config 딕셔너리 따라 적절한 Embedding 객체 생성 및 로딩 완료 후 반환.
    
    * 사용 예시:
        config = {"embedding": {"embed_model": "openai", ...}}
        embedding = create_embedding(config)
        model = embedding.get_model()
    
    Args:
        config: 프로젝트 설정. config["embedding"]["embed_type"] 사용.
    
    Returns:
        load() 완료된 BaseEmbedding 클래스 객체
        
    Raises:
        ValueError: 지원하지 않는 embed_type일 경우
    """
    embed_type = config["embedding"]["embed_type"].lower().strip()
    
    # 1) 지원 여부 검증 (딕셔너리 lookup)
    if embed_type not in _EMBEDDINGS:
        supported = list(_EMBEDDINGS.keys())
        raise ValueError(
            f"❌ 지원하지 않는 임베딩 타입: '{embed_type}'. "
            f"지원되는 타입: {supported}"
        )
    
    # 2) 클래스 가져와서 인스턴스 생성
    embedding_class = _EMBEDDINGS[embed_type]
    embedding = embedding_class()
    
    # 3) 로딩 실행
    embedding.load(config)
    
    return embedding