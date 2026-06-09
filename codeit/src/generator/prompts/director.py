"""
director.py - PromptDirector 클래스(Director 역할)

빌더 패턴(Builder Pattern)의 'Director' = 조립 순서 지시하는 감독관.

* 왜 Director가 필요한가?
   - Builder만 쓰면 호출자가 직접 순서 알아야 함
   - Director가 "표준 조립 순서" 알고 있어서 호출자는 그냥 명령만 내림
   - 같은 Builder로 여러 종류의 프롬프트(RAG용, 요약용 등) 만들 수 있음

* 사용 예시:
   director = PromptDirector(PromptBuilder())
   rag_prompt = director.make_rag_prompt(question, docs, ...)
   summary_prompt = director.make_summary_prompt(history_text)

* 파이썬 디자인 패턴 -> 생성 패턴 -> 빌더 패턴 (Builder Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import List, Optional
# from langchain.schema import Document
from langchain_core.documents import Document

from src.generator.prompts.prompt import Prompt
from src.generator.prompts.builder import PromptBuilder

class PromptDirector:
    """
    프롬프트 조립 순서 지시하는 감독관.
    
    * 빌더 패턴(Builder Pattern)의 Director 클래스에 대응:
       - 작업자(builder) 한 명 배정받음 (__init__)
       - make_xxx_prompt() 호출 및 표준 조립 순서 지시
    """
    
    def __init__(self, builder: PromptBuilder) -> None:
        """감독관은 지시할 작업자(builder) 한 명 배정받음."""
        self.builder = builder
    
    def make_rag_prompt(
        self,
        question: str,
        documents: List[Document],
        chat_history: Optional[str] = None,
        template_name: str = "rfp_korean",
        include_source: bool = True,
    ) -> Prompt:
        """
        RAG 답변용 프롬프트 조립 명령.
        
        * 표준 조립 순서:
           "먼저 질문, 그 다음 컨텍스트, 그 다음 대화 내역, 마지막 템플릿!"
        
        Args:
            question: 사용자 질문
            documents: 검색된 문서 청크 리스트
            chat_history: 이전 대화 요약 (선택)
            template_name: 사용할 템플릿 이름 (기본 'rfp_korean')
            include_source: 출처 정보 포함 여부
        
        Returns:
            완성된 Prompt 객체
        """
        return (
            self.builder
            .reset()                                             # 빈 Prompt 준비
            .with_question(question)                             # 1️⃣ 질문
            .with_context(documents, include_source)             # 2️⃣ 문서
            .with_chat_history(chat_history)                     # 3️⃣ 대화
            .with_template(template_name)                        # 4️⃣ 템플릿
            .build()                                             # ✨ 완성!
        )
    
    def make_summary_prompt(
        self,
        history_text: str,
    ) -> Prompt:
        """
        대화 요약용 프롬프트 조립 명령.
        
        * 같은 Builder를 다른 순서/조합으로 사용 → 다른 종류의 Prompt 생산!
        
        Args:
            history_text: 요약할 대화 내역 텍스트
        
        Returns:
            완성된 Prompt 객체
        """
        return (
            self.builder
            .reset()
            .with_question(history_text)              # 요약 대상 텍스트를 question 자리에
            .with_context([], include_source=False)   # 컨텍스트 없음
            .with_chat_history(None)                  # 대화 내역 없음
            .with_template("chat_summarizer")
            .get_result()
        )
        
    # TODO: 필요 시 아래 주석친 함수 추가 구현 예정(2026.06.08 minjae)
    # def make_question_rewrite_prompt(self, ...) -> Prompt: ...
    # def make_evaluation_prompt(self, ...) -> Prompt: ...
    # def make_followup_prompt(self, ...) -> Prompt: ...