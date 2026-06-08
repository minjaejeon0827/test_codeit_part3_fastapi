"""
prompt.py - Prompt 클래스 (Product 역할)

빌더 패턴(Builder Pattern)에서 'Product' 역할을 담당하는 클래스.
빌더가 조립한 최종 결과물(완성된 프롬프트) 표현.

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)

class Prompt:
    """
    완성된 프롬프트 결과물.
    
    * 빌더 패턴 (Builder Pattern) Product 클래스에 대응:
       - 부품(parts)을 차곡차곡 담는 빈 상자 역할
       - show() 메서드로 내용 확인 가능
    """
    
    def __init__(self) -> None:
        # 조립된 부품들을 담을 딕셔너리
        self.parts: Dict[str, Any] = {
            "question": None,
            "context": "",
            "chat_history": "",
            "template": None,
            "metadata": {},
        }
        # 최종 조립된 프롬프트 문자열
        self.final_text: str = ""
    
    def set_part(self, key: str, value: Any) -> None:
        """부품을 부품 상자에 추가/갱신."""
        self.parts[key] = value
    
    def set_final_text(self, text: str) -> None:
        """최종 조립된 프롬프트 문자열을 저장."""
        self.final_text = text
        # 아래 주석친 체이닝 지원 기능 필요 시 사용
        # return self  
    
    def show(self) -> None:
        """완성된 프롬프트에 어떤 부품이 들어있는지 출력 (디버깅용)."""
        logger.info("=" * 50)
        logger.info("📦 Prompt 부품 목록:")
        for key, value in self.parts.items():
            preview = str(value)[:80] + "..." if len(str(value)) > 80 else value
            logger.info(f"  - {key}: {preview}")
        logger.info(f"  - final_text 길이: {len(self.final_text)} 글자")
        logger.info("=" * 50)
    
    def __str__(self) -> str:
        """str(prompt) 시 최종 텍스트 반환."""
        return self.final_text
    
    # 아래 주석친 코드 필요시 사용(2026.06.08 minjae)
    # 아래 __repr__ 호출 방법
    # >>> prompt = Prompt()
    # >>> print(prompt)              # __str__ 호출 → 최종 텍스트
    # >>> prompt                      # __repr__ 호출 → 디버그 정보 ⭐
    # Prompt(question='이 사업 예산은?', context_len=1024, final_text_len=2048)
    # def __repr__(self) -> str:
    #     """
    #     디버깅용 표현 (REPL/print에서 자동 호출).
    #     🆕 추가: 파이썬 표준 방식의 디버깅 출력.
    #     """
    #     question = self.parts.get("question")
    #     return (
    #         f"Prompt("
    #         f"question={question!r}, "
    #         f"context_len={len(self.parts.get('context', ''))}, "
    #         f"final_text_len={len(self.final_text)}"
    #         f")"
    #     )