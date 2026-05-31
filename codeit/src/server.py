"""
웹서버 파일

* 파이썬 패키지 설치 명령어
pip install streamlit==1.52.2
pip install fastapi==0.104.1
pip install uvicorn==0.27.0.post1

* fastapi 웹서버 터미널 실행 명령어
uvicorn src.server:app --reload

* Gemini AI 도구 활용
참고: https://gemini.google.com/app/b27729891de22455?hl=ko
"""

from src.utils.cache_setup import setup_cache_dirs 
setup_cache_dirs()  # 먼저 호출!

from fastapi import FastAPI
# from src.generator.hf_generator import load_hf_model
# from src.generator.openai_generator import load_openai_model


import logging
logger = logging.getLogger(__name__)

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