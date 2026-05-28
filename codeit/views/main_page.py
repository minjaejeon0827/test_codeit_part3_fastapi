"""
streamlit 메인 웹페이지.

* streamlit 라이브러리 사용법 위키독스
참고: https://wikidocs.net/231560

* streamlit 메인 웹페이지 -> 서브 웹페이지 이동
참고: https://leemcse.tistory.com/entry/%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99-main%EA%B3%BC-sub-%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99

* Gemini AI 도구 활용
참고: https://gemini.google.com/app/b27729891de22455?hl=ko
"""

import streamlit as st
import requests

def main_page():
    """Streamlit 메인 웹페이지"""
    try:
        # Streamlit 페이지 설정
        st.set_page_config(
            page_title="[테스트] 2026-LLM-Project: RFP Summarizer & QA Chatbot", 
            layout="wide"
        )

        st.header("[테스트] RFPilot", divider='blue')
        st.caption("PDF나 한글(HWP) 제안서를 올리면, 챗봇이 핵심만 요약해주고 궁금한 점도 답변해 드려요!")

        
        # 간단한 푸터
        st.markdown("---")
        st.caption("© 2026 Codeit-Part3-5Team. All rights reserved.")
    except Exception as e:
        st.error(f"[오류] 기능 실행 중 오류 발생: {str(e)}")
        # st.stop()  # 아래 코드 실행 안 함

if __name__ == "__main__":
    main_page()