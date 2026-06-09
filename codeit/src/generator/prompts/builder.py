"""
builder.py - PromptBuilder 클래스(Builder 역할)

빌더 패턴(Builder Pattern) 'Builder' = 부품을 실제로 조립하는 작업자.

* 사용 예시 (Fluent Builder 스타일):
    prompt = (
        PromptBuilder()
        .reset()
        .with_question("이 사업 예산은?")
        .with_context(retrieved_docs)
        .with_chat_history(prev_history)
        .with_template("rfp_korean")
        .build()
    )

* 파이썬 디자인 패턴 -> 생성 패턴 -> 빌더 패턴 (Builder Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

from typing import List, Optional
# from langchain.schema import Document
from langchain_core.documents import Document

from src.generator.prompts.prompt import Prompt
from src.generator.prompts.templates import get_template

class PromptBuilder:
    """
    프롬프트 단계별로 조립하는 빌더(작업자).
    
    * 빌더 패턴(Builder Pattern) Builder 클래스에 대응:
       - __init__ 생성자 호출 및 Prompt 클래스 객체 생성
       - with_xxx() 메서드 호출 및 부품 조립
       - build() 메서드 호출 및 완성품 반환
    
    * Fluent 인터페이스: 메서드 체이닝 지원.
        builder.with_question(...).with_context(...).with_template(...)
    """
    
    def __init__(self):
        """생성자 호출 및 Prompt 클래스 객체 생성"""
        self.prompt = Prompt()
        self._include_source = True
    
    def reset(self) -> "PromptBuilder":
        """Prompt 클래스 객체 생성 및 리셋"""
        self.prompt = Prompt()
        self._include_source = True
        return self

    # ============================================================
    # (with_ 접두사: fluent builder 스타일- 실전 파이썬 컨벤션)
    # ============================================================

    def with_question(self, question: str) -> "PromptBuilder":
        """① 질문 추가/갱신."""
        self.prompt.set_part("question", question)
        return self
    
    def with_context(
        self, 
        documents: List[Document], 
        include_source: bool = True,
    ) -> "PromptBuilder":
        """② 검색된 문서들로부터 컨텍스트 추가/갱신."""
        self._include_source = include_source
        
        if not documents:
            self.prompt.set_part("context", "")
            return self
        
        context_blocks = []
        for chunk in documents:
            source_info = ""
            if include_source:
                source_info = self._format_source_info(chunk)
            block = f"{source_info}\n{chunk.page_content}".strip()
            context_blocks.append(block)
        
        context_text = "\n\n---\n\n".join(context_blocks)
        self.prompt.set_part("context", context_text)
        self.prompt.set_part("metadata", {
            "num_docs": len(documents),
            "include_source": include_source,
        })
        return self
    
    def with_chat_history(self, chat_history: Optional[str]) -> "PromptBuilder":
        """③ 이전 대화 추가/갱신."""
        self.prompt.set_part("chat_history", chat_history or "")
        return self
    
    def with_template(self, template_name: str) -> "PromptBuilder":
        """④ 사용할 템플릿 선택."""
        template_str = get_template(template_name)
        self.prompt.set_part("template", template_str)
        self.prompt.set_part("template_name", template_name)
        return self


    def build(self) -> Prompt:
        """
        모든 부품 끼워 맞춰서 완성된 Prompt 반환.
        
        * 빌더 패턴(Builder Pattern) '완성품 출하' 단계.
        
        Raises:
            ValueError: 필수 부품(question, template) 누락된 경우
        """
        # 필수 부품 검증
        if self.prompt.parts.get("question") is None:
            raise ValueError("❌ with_question() 호출 필수.")
        if self.prompt.parts.get("template") is None:
            raise ValueError("❌ with_template() 호출 필수.")
        
        # 템플릿에 부품 끼워넣기
        template = self.prompt.parts["template"]
        final_text = template.format(
            chat_history_section=self.prompt.parts["chat_history"],
            context=self.prompt.parts["context"],
            question=self.prompt.parts["question"],
        )
        
        self.prompt.set_final_text(final_text)
        return self.prompt
    
    # ============================================================
    # 내부 헬퍼 (private)
    # ============================================================
    def _format_source_info(self, chunk: Document) -> str:
        """청크 메타데이터에서 출처 정보 문자열 생성."""
        return (
            f"[출처: {chunk.metadata.get('파일명')} | "
            f"기관: {chunk.metadata.get('발주 기관')} | "
            f"사업명: {chunk.metadata.get('사업명')} | "
            f"청크 번호: {chunk.metadata.get('chunk_idx')}]"
        )