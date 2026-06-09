"""
chat_summary.py - 대화 요약 워크플로우

템플릿 메소드 패턴 (Template Method Pattern) 'ConcreteClass'에 대응.
부모(BaseWorkflow)가 비워둔 빈칸을 채워서 대화 요약 기능 구현.

* 부모(BaseWorkflow)가 정해둔 4단계 흐름 따르고,
  1단계(prepare_inputs)와 2단계(build_prompt)만 채워 넣음.

* 파이썬 디자인 패턴 -> 행위 패턴 -> 템플릿 메소드 패턴 (Template Method Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import logging
from typing import Any, Dict, List
from langsmith import traceable

# 내부 임포트
from src.generator.base import BaseGenerator  # 타입 힌트
from src.generator.workflows.workflow import BaseWorkflow
from src.generator.prompts.builder import PromptBuilder
from src.generator.prompts.director import PromptDirector
from src.generator.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

class ChatSummaryWorkflow(BaseWorkflow):
    """
    대화 요약 작업 흐름.
    
    * 템플릿 메소드 패턴 (Template Method Pattern) 'ConcreteClass'에 대응:
       - 부모가 정해둔 흐름은 그대로
       - operation1, operation2 빈칸만 채움
    """
    
    def prepare_inputs(self, chat_history_list: List[Dict]) -> Dict[str, Any]:
        """
        * operation1 빈칸 채우기에 대응:
           대화 리스트 -> 텍스트 변환.
        """
        history_text = "\n".join(
            f"{'질문' if turn['role'] == 'user' else '답변'}: {turn['content']}"
            for turn in chat_history_list
        )
        return {"history_text": history_text}
    
    def build_prompt(self, prepared_data: Dict[str, Any]) -> Prompt:
        """
        * operation2 빈칸 채우기에 대응:
           빌더 패턴(Builder Pattern)으로 요약용 프롬프트 조립.
        """
        director = PromptDirector(PromptBuilder())
        return director.make_summary_prompt(
            history_text=prepared_data["history_text"]
        )
    
    def log_result(self, answer: str) -> None:
        """대화 요약 결과 자세히 출력 (부모 메서드 오버라이드)."""
        # 변경: print → logger.info (부모 클래스와 일관성)
        logger.info(f"✅ 과거 대화 내역 요약: {answer}")


# ============================================================
# 외부 호출용 헬퍼 함수
# 함수명 변경: load_chat_history → summarize_chat_history
# ============================================================
@traceable(name="chat_summary_pipeline")
def summarize_chat_history(
    chat_history_list: List[Dict],
    generator: BaseGenerator,
    generation_config: Dict,
) -> str:
    """
    LLM 모델 이용하여 대화 내역 요약.
    
    * 변경 사항:
       - 함수 이름이 동작과 일치 (load → summarize)
       - if/else 모델 분기 사라짐 (전략 패턴 적용 결과)
       - 빌더 패턴으로 프롬프트 조립
       - 템플릿 메소드 패턴으로 흐름 표준화
    
    Args:
        chat_history_list: 대화 턴 리스트 [{"role": "user", "content": "..."}, ...]
        generator: 미리 로딩된 생성기 (전략 패턴)
        generation_config: 생성 설정 (max_length 등)
    
    Returns:
        요약된 대화 내역 (비어있으면 빈 문자열)
    """
    if not chat_history_list:
        return ""
    
    workflow = ChatSummaryWorkflow(generator, generation_config)
    return workflow.run(chat_history_list=chat_history_list)

# TODO: 아래 주석친 코드 필요 시 사용(2026.06.09 minjae)
# 변경: 기존 이름 호환성을 위한 별칭 (점진적 마이그레이션용)
# load_chat_history = summarize_chat_history