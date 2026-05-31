"""
cache_setup.py - ML 라이브러리 캐시 디렉토리 설정

이 모듈은 ML 라이브러리들의 모델 캐시 위치를 환경변수로 일관되게 지정.
앱 시작 시 단 1회, 모든 ML 관련 import 이전 호출 필수.
"""

import os
from pathlib import Path

def setup_cache_dirs() -> None:
    """
    ML 라이브러리들이 사용할 캐시 디렉토리를 환경변수로 설정.

    HuggingFace, Sentence-Transformers, Torch, LangChain 등은 
    모델/데이터를 다운로드해 로컬에 캐싱하는데, 이 함수는 그 위치를 일관되게 지정함.

    ⚠️ 호출 위치 매우 중요:
        반드시 ML 관련 라이브러리(transformers, torch, langchain 등) import **이전** 호출 필수! 
        환경변수는 라이브러리 import 시점에 한 번만 읽히므로, 호출이 늦으면 효과가 없음.

        ✅ 올바른 순서:
            from src.utils.cache_setup import setup_cache_dirs
            setup_cache_dirs()
            from transformers import AutoModel  # 이제 import OK

        ❌ 잘못된 순서:
            from transformers import AutoModel  # 이미 캐시 위치 결정됨
            setup_cache_dirs()  # 너무 늦음!

    설정되는 환경변수:
        - HF_HOME: HuggingFace 모델 캐시 루트 (transformers, datasets 공통)
        - SENTENCE_TRANSFORMERS_HOME: Sentence-Transformers 모델 캐시
        - TORCH_HOME: PyTorch 모델 캐시 (torchvision 등)
        - LANGCHAIN_CACHE: LangChain 자체 캐시

    Note:
        - 환경변수는 프로세스 단위로 격리. 
          여러 프로세스(예: FastAPI 서버, Streamlit 앱)가 ML 라이브러리를 쓰면 각 프로세스에서 호출 필수.
        - 캐시 경로는 사용자 홈 디렉토리(~/.cache/...) 기준.
          Docker 또는 권한 제한 환경에서는 별도 설정 필요할 수 있음.
    """
    home_cache = Path.home() / ".cache"

    # HuggingFace 통합 캐시 (transformers, datasets, hub 모두 이걸 봄)
    os.environ["HF_HOME"] = str(home_cache / "huggingface")

    # Sentence-Transformers 전용 캐시
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(home_cache / "sbert")

    # PyTorch 모델 캐시 (torchvision 등 포함)
    os.environ["TORCH_HOME"] = str(home_cache / "torch")

    # LangChain 자체 캐시
    os.environ["LANGCHAIN_CACHE"] = str(home_cache / "langchain")
    
    # 아래 주석 처리된 코드 필요 시 참고(2026.05.31 minjae)
    # os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
    # os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/sbert')
    # os.environ['XDG_CACHE_HOME'] = os.path.expanduser('~/.cache/xdg')
    # os.environ['LANGCHAIN_CACHE'] = os.path.expanduser('~/.cache/langchain')
    # os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

    # # 문제시 삭제
    # os.environ["HF_HOME"] = os.path.abspath("2026-LLM-Project/.cache")