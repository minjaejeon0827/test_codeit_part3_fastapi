"""
hf_generator.py - HuggingFace 모델 기반 답변 생성기

BaseGenerator 클래스 상속받아 HuggingFace 트랜스포머 모델로 
답변 생성하는 전략 패턴 (Strategy Pattern) 기반 클래스.

전략 패턴 (Strategy Pattern)에서의 위치:
    BaseGenerator (Strategy 인터페이스)
        ├── HFGenerator       ← 이 파일 (구체적 전략 1)
        ├── OpenAIGenerator   ← (구체적 전략 2)
        ├── ClaudeGenerator   ← (미래 추가 가능)
        └── GeminiGenerator   ← (미래 추가 가능)

파이썬 디자인 패턴 -> 행위 패턴 -> 전략 패턴 (Strategy Pattern)
참고: https://wikidocs.net/252293

* Claude AI 도구 활용
참고: https://claude.ai/chat/658340cd-271c-4cc0-8550-39c500607db3
"""

import os
from typing import Dict
from inspect import signature

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langsmith import trace

# 내부 임포트
from src.generator.base import BaseGenerator

class HFGenerator(BaseGenerator):
    """
    HuggingFace 트랜스포머 모델 사용한 답변 생성기.
    
    * 주요 기능:
      - HF Hub에서 모델/토크나이저 로딩
      - 4-bit 양자화 옵션 지원 (메모리 절약)
      - 답변 생성 + 자동 후처리
      - LangSmith 추적 통합
    
    * 사용 예시:
      >>> generator = HFGenerator()
      >>> generator.load(config)
      >>> answer = generator.generate("질문: ...", {"max_length": 512})
    """
    
    def __init__(self):
        """
        초기화 시 모델/토크나이저는 None으로 시작.
        실제 로딩은 load() 호출 시 실행.
        
        Note:
            이렇게 분리하면 객체 생성 비용은 가볍고, 
            실제 무거운 작업은 명시적으로 load() 부를 때만 발생.
        """
        self.tokenizer = None
        self.model = None
        self.model_name = None
    
    def load(self, config: Dict) -> None:
        """
        HuggingFace 모델과 토크나이저 로딩.
        
        Args:
            config: 설정 딕셔너리. 다음 키 필요:
                - generator.model_name (str): 모델 이름 (예: 'Markr-AI/Gukbap-Qwen2.5-7B')
                - generator.use_quantization (bool, optional): 4-bit 양자화 사용 여부
        
        Raises:
            EnvironmentError: 양자화 요청했는데 GPU 없을 때
        """
        self.model_name = config["generator"]["model_name"]
        use_quantization = config["generator"].get("use_quantization", False)
        
        # 1) 토크나이저 로딩 (질문/답변 → 숫자 변환기)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        
        # 2) 모델 로딩 (양자화 옵션에 따라 분기)
        if use_quantization:
            self.model = self._load_quantized_model()
        else:
            self.model = self._load_standard_model()
        
        # 3) 평가 모드 전환 (학습 모드 비활성화 → 추론 최적화)
        self.model.eval()
    
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
            RuntimeError: load()를 먼저 호출하지 않은 경우
        """
        # 안전장치: 모델 로딩 안 됐으면 에러
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "❌ HFGenerator.generate(): load() 먼저 호출해주세요."
            )
        
        # LangSmith 추적 시작 (디버깅용 발자국 기록)
        with trace(name="HFGenerator.generate", inputs={"prompt": prompt}) as run:
            
            # 1) 프롬프트를 토큰으로 변환
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            # 2) 생성 옵션 구성
            generate_kwargs = self._build_generate_kwargs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=generation_config.get("max_length", 512),
            )
            
            # 3) 모델별 옵션 추가 (token_type_ids 지원 여부 체크)
            if "token_type_ids" in inputs and "token_type_ids" in signature(self.model.generate).parameters:
                generate_kwargs["token_type_ids"] = inputs["token_type_ids"].to(self.model.device)
            
            # 4) 실제 생성 실행 (그래디언트 계산 비활성화로 메모리 절약)
            with torch.no_grad():
                output = self.model.generate(**generate_kwargs)
            
            # 5) 결과 디코딩 (프롬프트 부분은 제외하고 생성된 부분만)
            answer = self._decode_output(output, input_ids)
            
            # 6) 공통 후처리 (BaseGenerator에서 상속)
            answer = self._post_process(answer)
            
            # 7) 추적 결과 기록
            run.add_outputs({"output": answer})
            
            return answer
    
    # ============================================================
    # 내부 헬퍼 메서드 (private, 외부에서 직접 호출 X)
    # ============================================================
    
    def _load_quantized_model(self) -> AutoModelForCausalLM:
        """
        4-bit 양자화 적용된 모델 로딩.
        
        Returns:
            양자화된 HuggingFace 모델
        
        Raises:
            EnvironmentError: GPU 없을 때 (양자화는 GPU 필수)
        """
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "❌ GPU가 사용 불가능합니다. 양자화 모델은 GPU가 필요합니다. "
                "use_quantization=False로 설정하거나 GPU 환경에서 실행하세요."
            )
        
        # bitsandbytes 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                          # 4-bit로 양자화
            bnb_4bit_use_double_quant=True,             # 이중 양자화 (더 정확)
            bnb_4bit_quant_type="nf4",                  # 정규 분포 4-bit
            bnb_4bit_compute_dtype=torch.float16,       # 계산은 float16
            llm_int8_enable_fp32_cpu_offload=True,      # CPU 오프로딩 허용
        )
        
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            device_map="auto",
        )
    
    def _load_standard_model(self) -> AutoModelForCausalLM:
        """양자화 없이 모델 로딩 (메모리 충분할 때)."""
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            device_map="auto",
        )
    
    def _build_generate_kwargs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
    ) -> Dict:
        """
        모델.generate()에 전달할 옵션 딕셔너리 구성.
        
        Note:
            결정론적(deterministic) 생성을 위해 do_sample=False, num_beams=1 설정.
            동일한 질문에 항상 같은 답변 → 재현성 보장.
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_length,
            "do_sample": False,             # 샘플링 비활성화 (재현성)
            "num_beams": 1,                  # 빔 서치 비활성화 (속도 우선)
            "temperature": None,             # 결정론적이라 무의미
            "top_k": None,
            "top_p": None,
            "eos_token_id": (
                self.tokenizer.eos_token_id 
                or self.tokenizer.pad_token_id
            ),
            "repetition_penalty": 1.2,       # 반복 페널티 (같은 단어 반복 억제)
        }
    
    def _decode_output(
        self,
        output: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> str:
        """
        모델 출력 텐서 문자열로 변환 (프롬프트 부분 제외).
        
        Args:
            output: model.generate() 결과 텐서
            input_ids: 입력 프롬프트 토큰
        
        Returns:
            생성된 답변 부분만 추출한 문자열
        """
        output_ids = output[0]
        input_len = input_ids.size(1)
        generated_ids = output_ids[input_len:]    # 프롬프트 이후 부분만!
        
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return generated_text.strip()