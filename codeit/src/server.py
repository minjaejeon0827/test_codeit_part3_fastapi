"""
웹서버 파일

파이썬 패키지 설치 명령어:
    pip install streamlit==1.52.2
    pip install fastapi==0.104.1
    pip install uvicorn==0.27.0.post1

업그레이드 필요 패키지:
    참고: pip install google-genai 패키지 설치 시 오류 발생해서 fastapi, uvicorn 패키지 업그레이드 처리(2026.06.05 minjae)
    1) FastAPI 업그레이드
    pip install --upgrade fastapi

    2) 함께 사용하는 uvicorn도 같이 업그레이드 (호환성 ↑)
    pip install --upgrade uvicorn

    3) 설치 확인
    pip show fastapi

* fastapi 웹서버 터미널 실행 명령어
uvicorn src.server:app --reload

* Gemini AI 도구 활용
참고: https://gemini.google.com/app/b27729891de22455?hl=ko
"""

from src.utils.cache_setup import setup_cache_dirs 
setup_cache_dirs()  # 먼저 호출!

from fastapi import FastAPI


from src.generator.model_loader import load_generation_model
# from src.generator.hf_generator import load_hf_model
# from src.generator.openai_generator import load_openai_model
from src.utils.config import load_config
from src.settings import PROJECT_ROOT

import logging
logger = logging.getLogger(__name__)

# 앱 시작 시 모델 1회 로딩
config = load_config(PROJECT_ROOT)
model_info = load_generation_model(
    config["generator"]["model_type"],
    config["generator"]["model_name"],
    config["generator"]["use_quantization"],
)

app = FastAPI(title="Codeit API Server")

@app.get("/")
async def root():
    """
    루트 엔드포인트: Streamlit 프론트엔드에서 서버 연결 상태 확인 시 사용.
    """
    try:
        # 실제 환경에서는 DB 연결 확인이나 GPU 상태 등 체크 가능.
        return {
            "status": "success",
            "message": "Codeit AI FastAPI 서버 정상 작동 중!",
            "version": "1.0"
        }
        
    except Exception as e:
        logger.error(f"❌ 서버 연결 상태 오류: {e}")
        return {"status": "error", "message": str(e)}
    
@app.post("/reload-model")
async def reload_model():
    global model_info
    try:
        return {"status": "success", "message": "[테스트] 모델 리로드 구현 예정"}
        
        # 아래 주석친 코드 추후 구현 예정(2026.05.31 minjae)
        # model_info = load_generation_model(
        #     config["generator"]["model_type"],
        #     config["generator"]["model_name"],
        #     config["generator"]["use_quantization"],
        # )
        # return {"status": "success", "message": "모델 리로드 완료"}
    except Exception as e:
        return {"status": "error", "message": str(e)}