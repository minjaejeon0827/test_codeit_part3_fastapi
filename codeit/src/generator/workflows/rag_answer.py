"""
rag_answer.py - RAG 답변 생성 워크플로우

템플릿 메소드 패턴 (Template Method Pattern) 또 다른 'ConcreteClass'.
같은 4단계 흐름을 따르지만, 빈칸은 RAG 답변에 맞게 채움.

* LangChain + LangGraph AI 에이전트 개발
참고: https://youtu.be/3My9sphTxtk?si=nFA9vXJOj6dGzi3b

* 파이썬 디자인 패턴 -> 행위 패턴 -> 템플릿 메소드 패턴 (Template Method Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import logging
from typing import Any, Dict, List, Optional
from langsmith import traceable
# from langchain.schema import Document
from langchain_core.documents import Document

# 내부 임포트
from src.generator.base import BaseGenerator
from src.generator.workflows.workflow import BaseWorkflow
from src.generator.prompts.builder import PromptBuilder
from src.generator.prompts.director import PromptDirector
from src.generator.prompts.prompt import Prompt

logger = logging.getLogger(__name__)

class RAGAnswerWorkflow(BaseWorkflow):
    """
    RAG 답변 생성 작업 흐름.
    
    * 같은 부모(BaseWorkflow) 클래스 상속하지만 빈칸을 다르게 채움
      → 코드 중복 없이 다양한 작업 처리!
    """
    
    def prepare_inputs(
        self,
        retrieved_docs: List[Document],
        query: str,
        chat_history: Optional[str] = None,
        template_name: str = "rfp_korean",
        include_source: bool = True,
    ) -> Dict[str, Any]:
        """
        * operation1 빈칸 채우기에 대응:
           RAG 작업에 필요한 데이터 묶음.
        """
        if not retrieved_docs:
            raise ValueError("❌ 검색된 문서가 없습니다.")
        
        return {
            "query": query,
            "documents": retrieved_docs,
            "chat_history": chat_history,
            "template_name": template_name,
            "include_source": include_source,
        }
    
    def build_prompt(self, prepared_data: Dict[str, Any]) -> Prompt:
        """
        * operation2 빈칸 채우기에 대응:
            빌더 패턴으로 RAG 프롬프트 조립.
        """
        director = PromptDirector(PromptBuilder())
        return director.make_rag_prompt(
            question=prepared_data["query"],
            documents=prepared_data["documents"],
            chat_history=prepared_data["chat_history"],
            template_name=prepared_data["template_name"],
            include_source=prepared_data["include_source"],
        )
    
    def log_result(self, answer: str) -> None:
        """답변 결과 로깅 (부모 메서드 오버라이드)."""
        # 변경: print → logger.info
        logger.info(f"✅ 답변: {answer[:200]}...")
        logger.info(f"✅ 답변 생성 완료")


# ============================================================
# 외부 호출용 헬퍼 함수
# 함수명 변경: generator_main → run
# ============================================================
@traceable(name="rag_answer_pipeline")
def run(
    retrieved_docs: List[Document],
    query: str,
    generator: BaseGenerator,
    generation_config: Dict,
    chat_history: Optional[str] = None,
    template_name: str = "rfp_korean",
    include_source: bool = True,
) -> str:
    """
    RAG 답변 생성 파이프라인 실행.
    
    * 변경 사항:
        - 이름이 embedding/pipeline.py의 run()과 대칭
        - @traceable로 LangSmith 자동 추적
        - if/else 모델 분기 사라짐 (전략 패턴)
        - 빌더 패턴으로 프롬프트 조립
        - 템플릿 메소드 패턴으로 흐름 표준화
    
    Args:
        retrieved_docs: 검색된 문서 청크 리스트
        query: 사용자 질문
        generator: 미리 로딩된 생성기 (전략 패턴)
        generation_config: 생성 설정 (max_length 등)
        chat_history: 이전 대화 요약 (선택)
        template_name: 사용할 프롬프트 템플릿 이름
        include_source: 출처 정보 포함 여부
    
    Returns:
        생성된 답변 문자열
    """
    workflow = RAGAnswerWorkflow(generator, generation_config)
    return workflow.run(
        retrieved_docs=retrieved_docs,
        query=query,
        chat_history=chat_history,
        template_name=template_name,
        include_source=include_source,
    )
  
# TODO: 아래 주석친 코드 필요 시 사용(2026.06.09 minjae)
# 기존 이름 호환성을 위한 별칭 (점진적 마이그레이션)
# generator_main = run