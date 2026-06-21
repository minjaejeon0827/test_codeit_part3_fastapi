"""
gemini_generator.py - Google Gemini 모델 기반 답변 생성기

BaseGenerator 클래스 상속받아 Google GenAI API로 
답변 생성하는 전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴 (Strategy Pattern)에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator         (로컬 GPU)
        ├── OpenAIGenerator     (유료 API)
        ├── ClaudeGenerator     (유료 API)
        ├── GeminiGenerator     ← 이 파일 (무료 티어 있음)
        ├── GroqGenerator       (무료 + 초고속)
        └── OllamaGenerator     (무료, 로컬/원격 서버)

*  OpenAI/Claude와의 주요 차이점:
    - SDK 패키지: google-genai (2025년 신규 SDK, 이전 google-generativeai 대체)
    - API 메서드: client.models.generate_content()
    - 응답 구조: response.text
    - 설정 방식: GenerateContentConfig 객체 사용

설치 필요 패키지:
    pip install google-genai
    
파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
from typing import Dict

from google import genai
from google.genai import types
from langsmith import trace

from src.generator.base import BaseGenerator


class GeminiGenerator(BaseGenerator):
    """
    Google Gemini API 사용한 답변 생성기.
    
    *  주요 기능:
        - Google GenAI 클라이언트 초기화
        - Gemini 2.5 Pro/Flash 등 지원
        - 답변 생성 + 자동 후처리
        - LangSmith 추적 통합
    
    *  사용 예시:
        >>> generator = GeminiGenerator()
        >>> generator.load(config)
        >>> answer = generator.generate("질문: ...", {"max_length": 1024})
    
    *  지원 모델 예시:
        - gemini-2.5-pro
        - gemini-2.5-flash
        - gemini-2.0-flash
    """
    
    def __init__(self):
        self.client = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        Google GenAI 클라이언트 초기화.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'gemini-2.5-flash')
        
        Raises:
            ValueError: GOOGLE_API_KEY 환경변수 없을 때
        """
        # 1) API 키 검증
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY 설정되어 있지 않습니다. "
                ".env 파일 확인 및 환경변수 설정 필수."
            )
        
        # 2) 클라이언트 생성
        self.client = genai.Client(api_key=api_key)
        
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
                "❌ GeminiGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        with trace(name="GeminiGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) Generate Content API 호출 (Gemini는 config 객체 사용)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=generation_config.get("max_length", 1024),
                    temperature=0.0,
                ),
            )
            
            # 2) 응답에서 텍스트 추출 (Gemini는 response.text로 간단)
            answer = response.text.strip() if response.text else ""
            
            # 3) 공통 후처리
            answer = self._post_process(answer)
            
            # 4) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer