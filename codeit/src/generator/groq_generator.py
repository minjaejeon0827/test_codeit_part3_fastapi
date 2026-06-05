"""
groq_generator.py - Groq 클라우드 기반 답변 생성기

BaseGenerator 클래스 상속받아 Groq API로 답변 생성하는 
전략 패턴(Strategy Pattern) 기반 클래스.

*  Groq의 강점:
    - 초고속 추론 속도 (자체 LPU 하드웨어 사용, 토큰당 ~700+ tokens/sec)
    - 무료 티어 매우 넉넉함 (분당 30 요청, 일일 14,400 요청)
    - LLaMA 3.1, Mixtral 등 오픈소스 모델 무료 제공
    - OpenAI SDK와 거의 동일한 인터페이스 (학습 곡선 낮음)

전략 패턴 (Strategy Pattern)에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator         (로컬 GPU)
        ├── OpenAIGenerator     (유료 API)
        ├── ClaudeGenerator     (유료 API)
        ├── GeminiGenerator     (무료 티어 있음)
        └── GroqGenerator       ← 이 파일 (무료 + 초고속)

설치 필요 패키지:
    pip install groq
    
API 키 발급:
    https://console.groq.com/keys (무료 가입)

지원 모델 예시 (2026년 6월 기준):
    - llama-3.1-8b-instant          ← 사용자님 선택! (빠름, 무료)
    - llama-3.3-70b-versatile       (더 똑똑하지만 약간 느림)
    - mixtral-8x7b-32768            (긴 컨텍스트)
    - gemma2-9b-it                  (Google 오픈모델)

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
from typing import Dict

from groq import Groq
from langsmith import trace

from src.generator.base import BaseGenerator


class GroqGenerator(BaseGenerator):
    """
    Groq API 사용한 답변 생성기 (LLaMA 3.1 8B Instant 등).
    
    *  주요 기능:
        - Groq API 클라이언트 초기화 (API 키 검증 포함)
        - LLaMA 3.1, Mixtral 등 무료 오픈소스 모델 지원
        - 답변 생성 + 자동 후처리
        - LangSmith 추적 통합
    
    *  사용 예시:
        >>> generator = GroqGenerator()
        >>> generator.load(config)
        >>> answer = generator.generate("질문: ...", {"max_length": 512})
    
    *  주의사항:
        - 무료 티어 분당 요청 제한 있음 (RPM: 30, TPM: 30,000)
        - 토큰 한도 초과 시 429 에러 발생 → 재시도 로직 권장
    """
    
    def __init__(self):
        """
        초기화 시 클라이언트 None으로 시작.
        실제 API 클라이언트 생성은 load() 호출 시 일어남.
        """
        self.client = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        Groq 클라이언트 초기화.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'llama-3.1-8b-instant')
        
        Raises:
            ValueError: GROQ_API_KEY 환경변수 없을 때
        """
        # 1) API 키 검증
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GROQ_API_KEY 설정되어 있지 않습니다. "
                "https://console.groq.com/keys 에서 무료 발급 후 "
                ".env 파일에 GROQ_API_KEY=... 형태로 추가해주세요."
            )
        
        # 2) 클라이언트 생성 (Groq SDK는 OpenAI와 거의 동일한 인터페이스!)
        self.client = Groq(api_key=api_key)
        
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
                "❌ GroqGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        # LangSmith 추적 시작
        with trace(name="GroqGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) Chat Completions API 호출 (OpenAI와 동일한 인터페이스!)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_config.get("max_length", 512),
                temperature=0.0,  # 결정론적 출력 (같은 질문 → 같은 답변)
            )
            
            # 2) 응답에서 텍스트 추출 (OpenAI와 동일한 응답 구조)
            answer = response.choices[0].message.content.strip()
            
            # 3) 공통 후처리 (BaseGenerator에서 상속)
            answer = self._post_process(answer)
            
            # 4) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer