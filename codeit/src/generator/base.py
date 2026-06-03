"""
base.py - 모든 LLM 생성기 공통 인터페이스 (추상 클래스)

전략 패턴(Strategy Pattern) 'Strategy' 역할:
- 모든 생성기가 따라야 할 표준 규칙(공통 인터페이스) 정의
- HFGenerator, OpenAIGenerator, ClaudeGenerator, GeminiGenerator 등은 해당 클래스 상속받아 구현

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

"""
base.py - 모든 LLM 생성기의 공통 인터페이스 (추상 클래스)

전략 패턴(Strategy Pattern)의 'Strategy' 역할:
- 모든 생성기가 따라야 할 표준 규칙(공통 인터페이스) 정의
- HFGenerator, OpenAIGenerator 등은 해당 클래스를 상속받아 구현
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseGenerator(ABC):
    """
    모든 LLM 생성기 추상 베이스 클래스.
    
    🎯 사용 방법:
        1) 해당 클래스 상속받음
        2) load(), generate() 두 메서드 구현 필수!
    
    🌟 전략 패턴의 의미:
        - "답변을 생성한다"는 행위는 같지만,
        - "어떻게(HuggingFace? OpenAI?) 생성하느냐"는 자식 클래스가 결정
    """
    
    @abstractmethod
    def load(self, config: Dict) -> None:
        """
        모델 로딩 (자식 클래스 구현 필수!).
        
        Args:
            config: 설정 딕셔너리 (model_name, use_quantization 등 포함)
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, generation_config: Dict) -> str:
        """
        답변 생성 (자식 클래스 구현 필수!)).
        
        Args:
            prompt: 입력 프롬프트
            generation_config: 생성 설정 (max_length 등)
        
        Returns:
            생성된 답변 문자열
        """
        pass
    
    def _post_process(self, answer: str) -> str:
        """
        모든 생성기에서 공통으로 사용하는 답변 후처리 (선택 오버라이드).
        
        - 불필요한 반복 표현 제거
        - 너무 짧은 답변 대체
        
        Note:
            자식 클래스가 해당 메서드를 그대로 쓰거나 재정의 가능.
        """
        # 모델에서 나오는 stop words 처리
        stop_strings = ["```", "<|endoftext|>", "Human:", "human:", "###"]
        for stop_str in stop_strings:
            if stop_str in answer:
                answer = answer.split(stop_str)[0].strip()
        
        # 반복되는 표현 제거
        bad_tokens = [
            "하십시오", "하실 수", "알고 싶어요", "하는데 필요한",
            "것을", "한다", "하십시오.", "하시기 바랍니다",
        ]
        for token in bad_tokens:
            answer = answer.replace(token, "")
        
        # 너무 짧거나 의미 없는 경우 대체
        if len(answer) < 10 or answer.count(" ") < 3:
            answer = "해당 문서에서 질문에 대한 명확한 정보를 찾을 수 없습니다."
        
        return answer