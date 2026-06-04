"""
openai_generator.py - OpenAI 모델 기반 답변 생성기

BaseGenerator 클래스 상속받아 OpenAI Chat Completions API로 
답변 생성하는 전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator         (구체적 전략 1)
        ├── OpenAIGenerator     ← 이 파일 (구체적 전략 2)
        ├── ClaudeGenerator     (구체적 전략 3)
        └── GeminiGenerator     (구체적 전략 4)

설치 필요 패키지:
    pip install openai
    
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

"""
openai_generator.py - OpenAI 모델 기반 답변 생성기

BaseGenerator를 상속받아 OpenAI Chat Completions API로 
답변을 생성하는 전략(Strategy) 구현체입니다.

전략 패턴에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator         (구체적 전략 1)
        ├── OpenAIGenerator     ← 이 파일 (구체적 전략 2)
        ├── ClaudeGenerator     (구체적 전략 3)
        └── GeminiGenerator     (구체적 전략 4)
"""

import os
from typing import Dict

from openai import OpenAI
from langsmith import trace

from src.generator.base import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI Chat Completions API 사용한 답변 생성기.
    
    *  주요 기능:
        - OpenAI API 클라이언트 초기화 (API 키 검증 포함)
        - GPT-4, GPT-4.1-nano 등 다양한 모델 지원
        - 답변 생성 + 자동 후처리
        - LangSmith 추적 통합
    
    *  사용 예시:
        >>> generator = OpenAIGenerator()
        >>> generator.load(config)
        >>> answer = generator.generate("질문: ...", {"max_length": 512})
    """
    
    def __init__(self):
        """
        초기화 시 클라이언트는 None으로 시작.
        실제 API 클라이언트 생성은 load() 호출 시 일어남.
        """
        self.client = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        OpenAI 클라이언트 초기화.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'gpt-4.1-nano')
        
        Raises:
            ValueError: OPENAI_API_KEY 환경변수 없을 때
        """
        # 1) API 키 검증
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY 설정되어 있지 않습니다. "
                ".env 파일 확인 및 환경변수 설정 필수."
            )
        
        # 2) 클라이언트 생성 (HTTP 통신 객체)
        self.client = OpenAI(api_key=api_key)
        
        # 3) 사용할 모델 이름 저장
        self.model_name = config["generator"]["model_name"]
    
    def generate(self, prompt: str, generation_config: Dict) -> str:
        """
        주어진 프롬프트에 대한 답변 생성.
        
        Args:
            prompt: 입력 프롬프트
            generation_config: 생성 설정. 다음 키 사용:
                - max_length (int, optional): 최대 토큰 수 (기본 512)
        
        Returns:
            후처리된 답변 문자열
        
        Raises:
            RuntimeError: load() 먼저 호출하지 않은 경우
        """
        # 안전장치: 모델 로딩 안 됐으면 에러
        if self.client is None:
            raise RuntimeError(
                "❌ OpenAIGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        # LangSmith 추적 시작
        with trace(name="OpenAIGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) Chat Completions API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_config.get("max_length", 512),
                temperature=0.0,  # 결정론적 출력 (같은 질문 → 같은 답변)
            )
            
            # 2) 응답에서 텍스트 추출
            answer = response.choices[0].message.content.strip()
            
            # 3) 공통 후처리 (BaseGenerator에서 상속)
            answer = self._post_process(answer)
            
            # 4) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer
