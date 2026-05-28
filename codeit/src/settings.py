"""
웹 서버 전용 설정 파일
"""

from pathlib import Path

# 프로젝트 루트 디렉토리 설정
# __file__이 src/settings.py에 있으므로, parent.parent를 해야 프로젝트 루트가 됩니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKEND_DIR  = PROJECT_ROOT / "src"
FRONTEND_DIR = PROJECT_ROOT / "views"