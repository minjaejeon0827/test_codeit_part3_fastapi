"""
workflow.py - BaseWorkflow 추상 클래스 (Abstract 역할)

템플릿 메소드 패턴 (Template Method Pattern) 'AbstractClass'.
흐름은 고정, 빈칸은 상속 받는 자식 클래스가 채움.

* 모든 LLM 작업(대화 요약, RAG 답변)은 다음 4단계를 거침:
   1. 입력 데이터 준비  (자식이 구현 - prepare_inputs)
   2. 프롬프트 생성     (자식이 구현 - build_prompt)
   3. LLM 호출         (공통 - 부모가 처리 - _call_llm)
   4. 결과 후처리       (자식이 선택적으로 오버라이드 - post_process)

이 흐름은 부모가 통제 → "Don't call us, we'll call you" (할리우드 원칙)

* 파이썬 디자인 패턴 -> 행위 패턴 -> 템플릿 메소드 패턴 (Template Method Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.generator.base import BaseGenerator
from src.generator.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """
    모든 LLM 작업 흐름의 추상 베이스 클래스.
    
    * 템플릿 메소드 패턴 (Template Method Pattern) 'AbstractClass' 대응:
       - template_method() = run() (흐름 고정)
       - operation1, operation2 = prepare_inputs, build_prompt, post_process (빈칸)
    """
    def __init__(
        self, 
        generator: BaseGenerator, 
        generation_config: Dict[str, Any],
    ) -> None:
        """
        Args:
            generator: 미리 로딩된 생성기 (전략 패턴)
            generation_config: 생성 설정 (max_length 등)
        """
        self.generator = generator
        self.generation_config = generation_config
    
    # ============================================================
    # 템플릿 메소드 - 흐름 통제 (자식이 오버라이드 X)
    # ============================================================
    
    def run(self, **kwargs) -> str:
        """
        * 템플릿 메소드 패턴 (Template Method Pattern) 'AbstractClass' 대응:
        
        "누가 작업하든 무조건 1→2→3→4 순서로 진행해라!"
        라고 흐름을 강력하게 통제하는 감독관.
        """
        # 1단계: 입력 데이터 준비
        prepared_data = self.prepare_inputs(**kwargs)
        
        # 2단계: 프롬프트 생성
        prompt: Prompt = self.build_prompt(prepared_data)
        
        # 3단계: LLM 호출 (공통 처리)
        raw_answer = self._call_llm(prompt)
        
        # 4단계: 결과 후처리 (선택적)
        final_answer = self.post_process(raw_answer)
        
        # 5단계: 로깅 (선택적)
        self.log_result(final_answer)
        
        return final_answer
    
    # ============================================================
    # ⚠️ 자식이 반드시 채워야 하는 빈칸 (abstract)
    # ============================================================
    
    @abstractmethod
    def prepare_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        * 템플릿 메소드 패턴 (Template Method Pattern) operation1 대응.
        
        입력 데이터를 워크플로우에 맞게 가공.
        예: 대화 리스트 → 텍스트 변환, 또는 docs/query 묶기.
        """
        pass
    
    @abstractmethod
    def build_prompt(self, prepared_data: Dict[str, Any]) -> Prompt:
        """
        * 템플릿 메소드 패턴 (Template Method Pattern) operation2 대응.
        
        준비된 입력 데이터로부터 Prompt 객체 생성.
        보통 PromptDirector 사용.
        """
        pass
    
    # ============================================================
    # 🟢 공통 구현 (자식이 그대로 쓰거나, 필요 시 오버라이드)
    # ============================================================
    
    def _call_llm(self, prompt: Prompt) -> str:
        """LLM 호출 (모든 워크플로우 공통)."""
        return self.generator.generate(str(prompt), self.generation_config)
    
    def post_process(self, answer: str) -> str:
        """결과 후처리 (기본은 그대로 반환, 필요 시 자식이 오버라이드)."""
        return answer
    
    def log_result(self, answer: str) -> None:
        """로깅 (기본은 logger.info, 필요 시 자식이 오버라이드)."""
        logger.info(f"✅ {self.__class__.__name__} 완료")