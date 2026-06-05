"""
factory.py - 생성기 객체 생성 담당하는 팩토리

팩토리 메소드 패턴(Factory Method Pattern):
    - model_type 문자열만 받으면 적절한 Generator 객체 생성 및 반환
    - 호출자는 어떤 클래스가 생성되는지 신경 쓸 필요 없음
    - 새 모델 추가 시 _GENERATORS 딕셔너리 한 줄만 추가!

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import Dict

from src.generator.base import BaseGenerator
from src.generator.hf_generator import HFGenerator
from src.generator.openai_generator import OpenAIGenerator
from src.generator.claude_generator import ClaudeGenerator
from src.generator.gemini_generator import GeminiGenerator
from src.generator.groq_generator import GroqGenerator


# "모델 타입 문자열 → 클래스" 매핑표
_GENERATORS: Dict[str, type] = {
    "huggingface": HFGenerator,
    "openai": OpenAIGenerator,
    "claude": ClaudeGenerator,
    "gemini": GeminiGenerator,
    "groq": GroqGenerator,
}


def create_generator(config: Dict) -> BaseGenerator:
    """
    config에 따라 적절한 Generator 객체 생성 및 로딩까지 끝낸 후 반환.
    """
    model_type = config["generator"]["model_type"].lower()
    
    if model_type not in _GENERATORS:
        supported = list(_GENERATORS.keys())
        raise ValueError(
            f"❌ 지원하지 않는 모델 타입: '{model_type}'. "
            f"지원되는 타입: {supported}"
        )
    
    generator_class = _GENERATORS[model_type]
    generator = generator_class()
    generator.load(config)
    
    return generator