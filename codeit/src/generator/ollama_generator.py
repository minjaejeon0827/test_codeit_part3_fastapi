"""
ollama_generator.py - Ollama 로컬/원격 서버 기반 답변 생성기

BaseGenerator 클래스 상속받아 Ollama API로 답변 생성하는
전략 패턴(Strategy Pattern) 기반 클래스.

* Ollama 강점:
    - 완전 무료 (오픈소스 모델 로컬/원격 실행)
    - API 키 불필요
    - GCP 등 원격 리눅스 서버에 모델 올려두고 외부 IP로 접근 가능
    - llama3.2, qwen2.5, gemma2 등 다양한 오픈소스 모델 지원

전략 패턴 (Strategy Pattern)에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator         (로컬 GPU)
        ├── OpenAIGenerator     (유료 API)
        ├── ClaudeGenerator     (유료 API)
        ├── GeminiGenerator     (무료 티어 있음)
        ├── GroqGenerator       (무료 + 초고속)
        └── OllamaGenerator     ← 이 파일 (무료, 로컬/원격 리눅스 서버)

* OpenAI/Groq 주요 차이점:
    - API 키 불필요 (서버 주소만 있으면 됨)
    - base_url 주소로 원격 서버 지정 (.env 파일 OLLAMA_BASE_URL)
    - LangChain ChatOllama 사용 (invoke 메서드)

Ollama 서버 사전 준비 (GCP 리눅스 서버):
    1) Ollama 설치: curl -fsSL https://ollama.com/install.sh | sh
    2) 모델 받기: ollama pull llama3.2
    3) 외부 접근 허용 설정 (중요!):
       - 환경변수: OLLAMA_HOST=0.0.0.0:11434
       - GCP 방화벽에서 11434 포트 열기
    4) 서버 실행: ollama serve

설치 필요 패키지:
    pip install langchain-ollama

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
from typing import Dict

from langchain_ollama import ChatOllama
from langsmith import trace

from src.generator.base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """
    Ollama API 사용한 답변 생성기 (llama3.2 등).
    
    * 주요 기능:
        - Ollama 서버 연결 (로컬 또는 GCP 원격 서버)
        - llama3.2, qwen2.5 등 오픈소스 모델 무료 사용
        - 답변 생성 + 자동 후처리
        - LangSmith 추적 통합
    
    * 사용 예시:
        >>> generator = OllamaGenerator()
        >>> generator.load(config)
        >>> answer = generator.generate("질문: ...", {"max_length": 512})
    
    * 지원 모델 예시:
        - llama3.2  ← 팀 중급 프로젝트 선택! (GCP 리눅스 서버)
        - llama3.1
        - qwen2.5
        - gemma2
    
    * 주의사항:
        - 원격 서버 주소값은 .env 파일 OLLAMA_BASE_URL 항목 값으로 읽음 (환경마다 다름)
        - 원격 서버 사용 시 네트워크 지연 발생 가능
    """
    
    def __init__(self):
        """초기화 시 클라이언트는 None으로 시작."""
        self.client = None
        self.model_name = None
        self.base_url = None
    
    def load(self, config: Dict) -> None:
        """
        Ollama 클라이언트 초기화.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'llama3.2')
        
        Note:
            base_url 주소값은 .env 파일 OLLAMA_BASE_URL 항목 값으로 읽음.
            환경마다 서버 위치가 다르기 때문 (로컬 / GCP 원격 리눅스 서버 등).
            API 키는 불필요 (Ollama 인증 없음).
        """
        # 1) 사용할 모델 이름 저장
        self.model_name = config["generator"]["model_name"]
        
        # 2) 원격 서버 주소값 .env 파일에서 읽기 (GCP 외부 IP 주소값)
        #    - 환경마다 다름 (로컬 localhost / GCP 외부 IP 주소값)
        #    - 없으면 기본값 localhost 사용
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # 3) ChatOllama 클라이언트 생성 (API 키 불필요!)
        self.client = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,  # 결정론적 출력 (같은 질문 → 같은 답변)
        )
    
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
        # 안전장치: 클라이언트 로딩 안 됐으면 에러
        if self.client is None:
            raise RuntimeError(
                "❌ OllamaGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        # LangSmith 추적 시작
        with trace(name="OllamaGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) 최대 토큰 수 반영해 클라이언트 재구성
            #    (Ollama는 num_predict로 최대 토큰 제어)
            client = self.client.bind(
                num_predict=generation_config.get("max_length", 512),
            )
            
            # 2) 답변 생성 (invoke 호출)
            response = client.invoke(prompt)
            
            # 3) 응답에서 텍스트 추출 (ChatOllama는 .content로 접근)
            answer = response.content.strip() if response.content else ""
            
            # 4) 공통 후처리 (BaseGenerator에서 상속)
            answer = self._post_process(answer)
            
            # 5) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer