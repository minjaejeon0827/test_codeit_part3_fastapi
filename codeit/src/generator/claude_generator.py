"""
claude_generator.py - Anthropic Claude 모델 기반 답변 생성기

BaseGenerator 클래스 상속받아 Anthropic Messages API로 
답변 생성하는 전략 패턴 (Strategy Pattern) 기반 클래스.

*  OpenAI와 주요 차이점:
    - API 메서드: client.messages.create() (vs OpenAI: client.chat.completions.create())
    - 응답 구조: response.content[0].text (vs OpenAI: response.choices[0].message.content)
    - max_tokens는 필수 인자 (OpenAI는 선택)

설치 필요 패키지:
    pip install anthropic
    
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
from typing import Dict

from anthropic import Anthropic
from langsmith import trace

from src.generator.base import BaseGenerator

class ClaudeGenerator(BaseGenerator):
    """
    Anthropic Claude Messages API 사용한 답변 생성기.
    
    🎯 주요 기능:
        - Anthropic API 클라이언트 초기화
        - Claude Sonnet, Opus, Haiku 등 다양한 모델 지원
        - 답변 생성 + 자동 후처리
        - LangSmith 추적 통합
    
    🌟 사용 예시:
        >>> generator = ClaudeGenerator()
        >>> generator.load(config)
        >>> answer = generator.generate("질문: ...", {"max_length": 1024})
    
    📌 지원 모델 예시:
        - claude-opus-4-5
        - claude-sonnet-4-5
        - claude-haiku-4-5
    """
    
    def __init__(self):
        self.client = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        Anthropic 클라이언트 초기화.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'claude-sonnet-4-5')
        
        Raises:
            ValueError: ANTHROPIC_API_KEY 환경변수 없을 때
        """
        # 1) API 키 검증
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ ANTHROPIC_API_KEY 설정되어 있지 않습니다. "
                ".env 파일 확인 및 환경변수 설정 필수."
            )
        
        # 2) 클라이언트 생성
        self.client = Anthropic(api_key=api_key)
        
        # 3) 사용할 모델 이름 저장
        self.model_name = config["generator"]["model_name"]
    
    def generate(self, prompt: str, generation_config: Dict) -> str:
        """
        주어진 프롬프트에 대한 답변 생성.
        
        Args:
            prompt: 입력 프롬프트
            generation_config: 생성 설정. 다음 키 사용:
                - max_length (int, optional): 최대 토큰 수 (기본 1024)
        
        Returns:
            후처리된 답변 문자열
        
        Raises:
            RuntimeError: load() 먼저 호출하지 않은 경우
        """
        if self.client is None:
            raise RuntimeError(
                "❌ ClaudeGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        with trace(name="ClaudeGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) Messages API 호출 (Claude는 max_tokens가 필수!)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=generation_config.get("max_length", 1024),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            # 2) 응답에서 텍스트 추출 (Claude는 content가 리스트 구조)
            answer = response.content[0].text.strip()
            
            # 3) 공통 후처리
            answer = self._post_process(answer)
            
            # 4) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer
