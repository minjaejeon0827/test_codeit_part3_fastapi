"""
config.py - Pydantic 기반 설정 로딩 및 검증 모듈
pyyaml: YAML 파일 읽고 쓰기 용도

설치 필요 패키지:
    pip install pyyaml pydantic

AI 도구 활용: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
import logging
from typing import Literal, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ConfigDict,
    model_validator,
)

logger = logging.getLogger(__name__)


# ============================================================
# 1. 섹션별 설정 스키마 (config.yaml 구조와 1:1 매핑)
# ============================================================

class SettingsConfig(BaseModel):
    """[settings] 섹션 - 전역 설정"""
    model_config = ConfigDict(extra="forbid")  # 정의되지 않은 키는 오타로 간주하여 에러

    verbose: bool = False
    project_root: str = ""


class DataConfig(BaseModel):
    """[data] 섹션 - 문서 로딩/분할 설정"""
    model_config = ConfigDict(extra="forbid")

    folder_path: str = "data/raw/files"
    data_list_path: str = "data/processed/data_list.csv"
    top_k: int = Field(default=5, ge=1, le=100, description="1~100 사이")
    file_type: Literal["hwp", "pdf", "all"] = "all"
    apply_ocr: bool = False
    splitter: Literal["section", "recursive", "token"] = "section"
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=250, ge=0)

class EmbeddingConfig(BaseModel):
    """[embedding] 섹션 - 임베딩/벡터DB 설정"""
    model_config = ConfigDict(extra="forbid")

    embed_type: Literal["openai", "huggingface", "claude", "gemini", "groq"] = "openai"
    embed_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: str = "data/vector_store"


class RetrieverConfig(BaseModel):
    """[retriever] 섹션 - 검색기 설정"""
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    search_type: Literal["similarity", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1)
    rerank: bool = True
    rerank_top_k: int = Field(default=5, ge=1)


class GeneratorConfig(BaseModel):
    """[generator] 섹션 - LLM 생성기 설정"""
    model_config = ConfigDict(extra="forbid")

    model_type: Literal["huggingface", "openai"] = "huggingface"
    model_name: str = ""
    max_length: int = Field(default=512, ge=1)
    use_quantization: bool = True


class AppConfig(BaseModel):
    """최상위 통합 설정 - 경로 검사를 여기서 통합 처리"""
    model_config = ConfigDict(extra="forbid")

    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    chat_history: list = Field(default_factory=list)

    @model_validator(mode="after")
    def _resolve_and_check_paths(self):
        """
        project_root 기준 상대 경로 -> 절대 경로 변환
        - 모든 섹션이 채워진 후에 실행되므로 project_root 알 수 있음
        - 입력 경로 없으면 경고만 (실행은 계속)
        - 출력 경로(vector_store)는 자동 생성
        """
        root = self.settings.project_root or os.getcwd()

        def resolve(p: str) -> str:
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))

        # 1) 상대경로 → 절대경로 변환 (가장 중요!)
        self.data.folder_path = resolve(self.data.folder_path)
        self.data.data_list_path = resolve(self.data.data_list_path)
        self.embedding.vector_store_path = resolve(self.embedding.vector_store_path)

        # 2) 입력 경로 검사 (없으면 경고만, 에러 X)
        if not os.path.exists(self.data.folder_path):
            logger.warning(f"⚠️  [Config] data.folder_path가 없습니다: {self.data.folder_path}")
        if not os.path.exists(self.data.data_list_path):
            logger.warning(f"⚠️  [Config] data.data_list_path가 없습니다: {self.data.data_list_path}")

        # 3) 출력 경로 자동 생성
        os.makedirs(self.embedding.vector_store_path, exist_ok=True)

        return self


# ============================================================
# 2. 내부 헬퍼
# ============================================================

def _format_validation_error(err: ValidationError) -> str:
    """Pydantic 검증 에러를 사람이 읽기 쉬운 한글 메시지 변환"""
    lines = ["❌ [Config] 설정값 검증 실패:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"])
        msg = e.get("msg", "")
        lines.append(f"  - '{loc}' → {msg}")
    return "\n".join(lines)


# ============================================================
# 3. 공개 API
# ============================================================

def check_config(config: Union[dict, AppConfig]) -> AppConfig:
    """
    설정의 유효성 검사 및 검증된 AppConfig 객체 반환.

    Args:
        config: 검증할 설정 (dict 또는 AppConfig)

    Returns:
        AppConfig: 검증된 설정 객체

    Raises:
        ValidationError: 설정 유효하지 않을 경우
    """
    if isinstance(config, AppConfig):
        return config
    return AppConfig.model_validate(config)


def load_config(project_root: str) -> dict:
    """
    config.yaml 파일 로드 및 검증한 후 dict 반환.
    main_page.py 등 기존 코드와의 호환을 위해 dict 반환.

    Args:
        project_root: 프로젝트 루트 절대 경로

    Returns:
        dict: 검증된 설정 딕셔너리

    Raises:
        FileNotFoundError: config.yaml 없을 때
        yaml.YAMLError: YAML 문법 오류 시
        ValidationError: 설정값 검증 실패 시
    """
    config_path = os.path.join(project_root, "config.yaml")

    # 1) 파일 존재 검사
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ [File] 설정 파일을 찾을 수 없습니다: {config_path}"
        )

    # 2) YAML 파싱
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error(f"❌ [YAML] 설정 파일 파싱 오류: {e}")
        raise
    except (PermissionError, OSError) as e:
        logger.error(f"❌ [File] 파일 접근 오류: {e}")
        raise

    # 3) project_root 주입 (settings 섹션에 자동 추가)
    if not isinstance(raw_config.get("settings"), dict):
        raw_config["settings"] = {}
    raw_config["settings"]["project_root"] = project_root

    # 4) Pydantic 스키마 검증
    try:
        validated = AppConfig.model_validate(raw_config)
    except ValidationError as e:
        logger.error(_format_validation_error(e))
        raise

    # 5) dict 변환 및 반환
    config_dict = validated.model_dump()

    # 6) verbose 모드일 때 전체 설정 출력
    if validated.settings.verbose:
        logger.info("📄 [Verbose] 최종 설정 내용:\n%s",
                    yaml.dump(config_dict, allow_unicode=True, sort_keys=False))

    return config_dict


# ============================================================
# 4. (선택) 타입 안전 버전 - 필요 시 사용
# ============================================================

def load_config_typed(project_root: str) -> AppConfig:
    """
    타입 안전 버전: dict가 아닌 AppConfig 객체를 반환합니다.

    장점:
        - IDE 자동완성 (config.data.top_k 입력 시 추천 뜸)
        - 정적 타입 체크 (mypy/pyright)
        - 오타 시 즉시 에러

    기존 dict 기반 코드는 load_config() 함수 그대로 사용.
    """
    config_dict = load_config(project_root)
    return AppConfig.model_validate(config_dict)