"""
streamlit 메인 웹페이지.

설치 필요 패키지:
    pip install python-dotenv

참고
* streamlit 라이브러리 사용법 위키독스
참고: https://wikidocs.net/231560

* streamlit 메인 웹페이지 -> 서브 웹페이지 이동
참고: https://leemcse.tistory.com/entry/%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99-main%EA%B3%BC-sub-%ED%8E%98%EC%9D%B4%EC%A7%80-%EC%9D%B4%EB%8F%99

* Gemini AI 도구 활용
참고: https://gemini.google.com/app/b27729891de22455?hl=ko

* .env 파일과 환경변수 관리
참고: https://wikidocs.net/344431
"""

import os         # 파일 경로/환경변수 처리
import shutil     # 폴더 삭제 등 파일 시스템 조작
import yaml       # config.yaml 파일 파싱/출력
import uuid       # 고유 식별자(세션 ID) 생성
import requests   # FastAPI 백엔드에 HTTP 요청 보내기
# import torch      # PyTorch (GPU 관련 설정용)
import streamlit as st  # Streamlit UI 프레임워크

from dotenv import load_dotenv  # .env 파일에서 환경변수 로딩
from dataclasses import dataclass

# 내부 임포트
from src.settings import PROJECT_ROOT, RELOAD_URL, CHAT_URL
from src.utils.config import load_config  # config.yaml 로딩
from src.embedding.pipeline import generate_index_name  # 벡터DB 인덱스 이름 생성

@st.cache_data
def _load_config_cached():
    """
    config.yaml을 로드하고 검증한 결과 캐싱하여 반환.

    Streamlit은 사용자 인터랙션이 발생할 때마다 스크립트를 위에서부터 다시 실행.
    이 함수에 `@st.cache_data` 데코레이터 붙여서 최초 1회만 실제 로딩이 일어나고,
    이후 호출은 캐시된 결과 즉시 반환.

    Returns:
        dict: 검증된 설정 딕셔너리 (config.yaml의 내용)

    Note:
        - 함수명에 언더스코어(_) 붙여 모듈 내부 전용 표시.
        - config.yaml 파일 수정한 경우, Streamlit 우측 상단 메뉴
          "Clear cache" 또는 'C' 단축키로 캐시 비워야 변경 사항 반영.
        - 환경 변수나 시스템 상태에 따라 결과가 달라지는 경우, 
          캐시가 오래된 값을 반환할 수 있으니 주의 필요.

    See Also:
        src.utils.config.load_config: 실제 로딩 및 검증 로직
    """
    try:
        return load_config(PROJECT_ROOT)
    except Exception as e:
        st.error(f"❌ config.yaml 설정 파일 로드 실패: {e}")
        st.stop()

def api_key_verification(embed_model):
    if embed_model.strip().lower() == "openai":
        load_dotenv()
        openai_key = os.environ.get("OPENAI_API_KEY")

        if not openai_key:
            openai_key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                key=f"openai_api_key_special"
            )
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            else:
                st.warning("OpenAI 모델 API 키 입력 필수!")

@dataclass(frozen=True)
class VectorStorePaths:
    """
    벡터 저장소 관련 파일 경로 모음.
    
    config + session_id 기반으로 현재 사용 중인 DB의 파일/폴더 경로를 캡슐화.
    """
    db_type: str          # "faiss" 또는 "chroma"
    faiss_file: str       # .faiss 파일 경로
    metadata_file: str    # .pkl 파일 경로
    chroma_path: str      # Chroma 폴더 경로
    
    @property
    def exists(self) -> bool:
        """
        현재 db_type 기준으로 벡터 저장소가 디스크에 존재하는지 확인.
        
        FAISS는 .faiss와 .pkl 두 파일이 모두 있어야 True.
        Chroma는 폴더가 존재하면 True.
        """
        if self.db_type == "faiss":
            return os.path.exists(self.faiss_file) and os.path.exists(self.metadata_file)
        elif self.db_type == "chroma":
            return os.path.exists(self.chroma_path)
        return False


def get_vector_store_paths(config: dict, session_id: str) -> VectorStorePaths:
    """
    현재 설정 기반으로 벡터 저장소 경로들을 계산하여 반환합니다.
    
    이 함수는 '경로 계산'의 단일 진실의 원천(Single Source of Truth)입니다.
    경로가 필요한 모든 곳에서 이 함수를 호출하세요.
    
    Args:
        config: 프로젝트 설정 딕셔너리
        session_id: 현재 세션 식별자
    
    Returns:
        VectorStorePaths: 계산된 경로 모음 객체
    """
    vector_store_dir = os.path.join(PROJECT_ROOT, config["embedding"]["vector_store_path"])
    index_name = f"{generate_index_name(config)}_{session_id}"
    
    return VectorStorePaths(
        db_type=config["embedding"]["db_type"],
        faiss_file=os.path.join(vector_store_dir, f"{index_name}.faiss"),
        metadata_file=os.path.join(vector_store_dir, f"{index_name}.pkl"),
        chroma_path=os.path.join(vector_store_dir, index_name),
    )

def reset_vector_store(config: dict, session_id: str) -> None:
    """
    설정에 명시된 벡터 저장소(FAISS 또는 Chroma) 디스크에서 삭제.
    
    Args:
        config: 프로젝트 설정 딕셔너리 (embedding 섹션 사용)
        session_id: 현재 세션 식별자 (인덱스 이름의 일부로 사용)
    
    Note:
        - FAISS: .faiss + .pkl 파일 2개 삭제
        - Chroma: 폴더 전체 삭제
        - 파일/폴더가 없으면 안내 메시지만 출력 (에러 X)
    """
    paths = get_vector_store_paths(config, session_id)
    
    try:
        if paths.db_type == "faiss":
            if paths.exists:
                os.remove(paths.faiss_file)
                os.remove(paths.metadata_file)
                st.success("✅ FAISS DB 삭제 완료")
            else:
                st.info("FAISS 파일이 존재하지 않습니다.")
        
        elif paths.db_type == "chroma":
            if paths.exists:
                shutil.rmtree(paths.chroma_path)
                st.success("✅ Chroma DB 삭제 완료")
            else:
                st.info("Chroma 폴더가 존재하지 않습니다.")
        
        else:
            st.warning(f"⚠️ 지원하지 않는 DB 타입: {paths.db_type}")
    
    except Exception as e:
        st.error(f"❌ Vector DB 삭제 실패: {e}")
    
# 아래 주석친 코드 필요 시 참고(2026.05.31 minjae)
# def reset_vector_store(config: dict, session_id: str) -> None:
#     """
#     설정에 명시된 벡터 저장소(FAISS 또는 Chroma) 디스크에서 삭제.
    
#     Args:
#         config: 프로젝트 설정 딕셔너리 (embedding 섹션 사용)
#         session_id: 현재 세션 식별자 (인덱스 이름의 일부로 사용)
    
#     Note:
#         - FAISS: .faiss + .pkl 파일 2개 삭제
#         - Chroma: 폴더 전체 삭제
#         - 파일/폴더가 없으면 안내 메시지만 출력 (에러 X)
#     """
#     db_type = config["embedding"]["db_type"]
#     vector_store_dir = os.path.join(PROJECT_ROOT, config["embedding"]["vector_db_path"])
#     index_name = f"{generate_index_name(config)}_{session_id}"
    
#     try:
#         if db_type == "faiss":
#             faiss_file = os.path.join(vector_store_dir, f"{index_name}.faiss")
#             pkl_file = os.path.join(vector_store_dir, f"{index_name}.pkl")
            
#             if os.path.exists(faiss_file):
#                 os.remove(faiss_file)
#                 os.remove(pkl_file)
#                 st.success("✅ FAISS DB 삭제 완료")
#             else:
#                 st.info("FAISS 파일이 존재하지 않습니다.")
        
#         elif db_type == "chroma":
#             chroma_path = os.path.join(vector_store_dir, index_name)
            
#             if os.path.exists(chroma_path):
#                 shutil.rmtree(chroma_path)
#                 st.success("✅ Chroma DB 삭제 완료")
#             else:
#                 st.info("Chroma 폴더가 존재하지 않습니다.")
        
#         else:
#             st.warning(f"⚠️ 지원하지 않는 DB 타입: {db_type}")
    
#     except Exception as e:
#         st.error(f"❌ Vector DB 삭제 실패: {e}")
        
def reload_generation_model() -> None:
    """
    FastAPI server.py 소스파일에 모델 리로드 요청 전송 및 결과 UI 표시.
    
    Note:
        - server.py 소스파일 /reload-model 엔드포인트 함수 구현되어 있어야 함
        - 리로드는 수십 초~수 분 걸릴 수 있음 (사용자에게 사전 안내 필요)
        - 네트워크 오류 시 명확한 에러 메시지 표시
    """
    try:
        with st.spinner("모델 리로드 중... (수십 초 소요)"):
            response = requests.post(RELOAD_URL, timeout=300)  # 5분 타임아웃
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ {result.get('message', '모델 리로드 완료')}")
        else:
            st.error(f"❌ 리로드 실패: HTTP {response.status_code}")
    
    except requests.Timeout:
        st.error("❌ 리로드 시간 초과 (5분). 백엔드 상태 확인하세요.")
    except requests.ConnectionError:
        st.error("❌ 백엔드 서버에 연결할 수 없습니다.")
    except Exception as e:
        st.error(f"❌ 예기치 못한 오류: {e}")

config = _load_config_cached()
# 필요 시 아래 주석친 코드 사용 예정(2026.05.31 minjae)
# load_dotenv()     # .env 파일에서 환경변수 읽어옴 (OPENAI_API_KEY 등)
# Streamlit과 PyTorch 호환성 문제 해결용 워크어라운드 (경로 충돌 방지)
# torch.classes.__path__ = []



def main_page():
    """Streamlit 메인 웹페이지"""
    # Streamlit 페이지 설정
    st.set_page_config(
        page_title="[테스트] 2026-LLM-Project: RFP Summarizer & QA Chatbot", 
        layout="wide"
    )

    st.header("[테스트] RFPilot", divider='blue')
    st.caption("PDF나 한글(HWP) 제안서를 올리면, 챗봇이 핵심만 요약해주고 궁금한 점도 답변해 드려요!")

    try:
        if "chat_history" not in st.session_state:  # 세션 상태 초기화
            st.session_state.chat_history = []
        else: # 세션 상태가 존재하는 경우, chat_history를 초기화하지 않음
            st.session_state.chat_history = st.session_state.get("chat_history", [])
            config["chat_history"] = st.session_state.chat_history

        if "docs" not in st.session_state:
            st.session_state.docs = None

        if "session_id" not in st.session_state:
            # uuid 고유 번호는 -(hyphen)을 포함해 36자, 너무 긺으로 자르는 과정 추가
            st.session_state.session_id = str(uuid.uuid4())[:8]

        session_id = st.session_state.session_id
        
        # 경로 정보 획득
        paths = get_vector_store_paths(config, session_id)
        
        # 사이드바 구성
        with st.sidebar:
            st.subheader("⚙️ 설정")
            # Data 관련 설정
            st.subheader("📂 데이터 설정")
            config["data"]["top_k"] = st.slider("🔢 최대 문서 수(files)", 1, 100, config["data"]["top_k"])
            config["data"]["file_type"] = st.selectbox("📄 파일 유형", ["all", "pdf", "hwp"], index=["all", "pdf", "hwp"].index(config["data"]["file_type"]))
            config["data"]["apply_ocr"] = st.toggle("🧾 OCR 적용 여부", config["data"]["apply_ocr"])
            config["data"]["splitter"] = st.selectbox("✂️ 문서 분할 방법", ["section", "recursive", "token"], index=["section", "recursive", "token"].index(config["data"]["splitter"]))
            config["data"]["chunk_size"] = st.number_input("📏 Chunk 크기", value=config["data"]["chunk_size"], step=100)
            config["data"]["chunk_overlap"] = st.number_input("🔁 Chunk 오버랩", value=config["data"]["chunk_overlap"], step=10)

            # Embedding 설정
            st.subheader("🧠 임베딩 설정")
            config["embedding"]["embed_model"] = st.text_input("🧬 임베딩 모델", config["embedding"]["embed_model"])
            config["embedding"]["db_type"] = st.selectbox("💾 Vector DB 타입", ["faiss", "chroma"], index=["faiss", "chroma"].index(config["embedding"]["db_type"]))

            # api_key 확인
            api_key_verification(config["embedding"]["embed_model"])

            # Retriever 설정
            st.subheader("🔍 리트리버 설정")
            config["retriever"]["search_type"] = st.selectbox("🔎 검색 방식", ["similarity", "hybrid"], index=["similarity", "hybrid"].index(config["retriever"]["search_type"]))
            config["retriever"]["top_k"] = st.slider("📄 검색 문서 수(chunks)", 1, 20, config["retriever"]["top_k"])
            config["retriever"]["rerank"] = st.toggle("📊 리랭크 적용", config["retriever"]["rerank"])
            config["retriever"]["rerank_top_k"] = st.slider("🔝 리랭크 문서 수(chunks)", 1, 20, config["retriever"]["rerank_top_k"])

            # Generator 설정
            st.subheader("🔍 생성자 설정")
            config["generator"]["model_type"] = st.selectbox("🔎 생성 모델 타입", ["huggingface", "openai"], index=["huggingface", "openai"].index(config["generator"]["model_type"]))
            config["generator"]["model_name"] = st.text_input("🧬 생성 모델", config["generator"]["model_name"])
            config["generator"]["max_length"] = st.number_input("🔢 최대 토큰 수(max_length)", value=config["generator"]["max_length"], step=32)

            # api_key 재확인
            api_key_verification(config["generator"]["model_type"])
            
            if st.button("⚠️ Vector DB 초기화"):
                reset_vector_store(config, session_id)
                    
            # 설정 버튼
            cols = st.columns([4, 6])
            with cols[0]:
                if st.button("🔄 리셋"):
                    st.session_state.chat_history = []
                    st.session_state.docs = None
                    st.session_state.past_chunks = []
                    st.rerun()
            with cols[1]:
                if st.button("🔁 모델 리로드"):
                    reload_generation_model()
                    st.rerun()

        # 탭 구성
        tab1, tab2 = st.tabs(["💬 챗봇", "📄 문서 요약 및 분석"])

        model_type = config["generator"]["model_type"]
        model_name = config["generator"]["model_name"]
        use_quantization = config["generator"]["use_quantization"]

        print("\n📄 [Verbose] 최종 설정 내용:")
        print(yaml.dump(config, allow_unicode=True, sort_keys=False))

        with tab1:
            query = st.chat_input("질문을 입력하세요")

            # 질문 처리
            if query:
                if not isinstance(query, str) or query.strip() == "":
                    st.warning("질문을 올바르게 입력해주세요.")
                    st.stop()

                # 사이드바 설정 반영 - Vector DB 존재 여부 확인
                if config["data"]["top_k"] == 100:
                    is_save = not paths.exists   # ← paths.exists property가 db_type별 분기 처리
                else:
                    is_save = True
                
                # 또는 한 줄로:
                # is_save = config["data"]["top_k"] != 100 or not paths.exists
                
                # 아래 주석친 코드 필요 시 참고(2026.05.31 minjae)
                # 사이드바 설정 반영 - Vector DB 존재 여부 확인
                # if config["data"]["top_k"] == 100:
                #     if config["embedding"]["db_type"] == "faiss":
                #         is_save = not os.path.exists(faiss_file)
                #     elif config["embedding"]["db_type"] == "chroma":
                #         is_save = not os.path.exists(chroma_path)
                #     else:
                #         is_save = True
                # else:
                #     is_save = True
                    
                # 질문 입력시 이전 추출문서 기록 초기화
                if st.session_state.docs is not None:
                    st.session_state.docs = None
                
                with st.chat_message("user"):
                    st.markdown(query)

                try:
                    with st.spinner("🤖 답변 생성 중..."):
                        # 아래 주석친 코드 추후 구현 예정(2026.05.31 minjae)
                        # st.info("❗ [테스트] 답변 생성 기능 추후 구현 예정!")
                        # response = requests.post(
                        #     CHAT_URL,
                        #     json={
                        #         "query": query,
                        #         "chat_history": st.session_state.chat_history,
                        #         "session_id": st.session_state.session_id,
                        #         "config": config
                        #     }
                        # )
                        
                        # if response.status_code != 200:
                        #     st.error(f"❌ API 요청 실패: {response.status_code} - {response.text}")
                        #     st.stop()
                            
                        # result = response.json()
                        # answer = result["answer"]
                        # elapsed = result["elapsed"]
                        # docs = result["docs"]
                        answer="[테스트] 답변 생성 기능 추후 구현 예정!"
                        elapsed = 1.0
                        docs = ["[테스트] 답변 생성 기능 추후 구현 예정!"]

                    # 결과 Streamlit에 반영
                    st.session_state.docs = docs 
                    
                    # 대화 이력 업데이트
                    st.session_state.chat_history.append({"role": "user", "content": query})
                    st.session_state.chat_history.append({"role": "ai", "content": answer})
                    
                    # 대화 기록 업데이트
                    config["chat_history"] = st.session_state.chat_history
                    
                    # 추론 시간 표시
                    with st.chat_message("assistant"):
                        st.markdown(f"🕒 **추론 시간:** {elapsed}초")
                        
                    # 랜더링 한계점: 20개까지 히스토리 표시
                    MAX_CHAT_HISTORY = 20
                    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                        st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류 발생: {e}")
                    st.stop()

            # 이전 대화 출력
            for turn in st.session_state.chat_history[::-1]:
                with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                    st.markdown(turn["content"])

        with tab2:
            st.subheader("📄 문서 요약 및 분석")

            docs = st.session_state.get("docs", None)

            if docs is None:
                st.info("❗ 먼저 질문을 입력하고 문서를 검색하세요.")
            elif isinstance(docs, list) and len(docs) > 0:
                for i, doc in enumerate(docs):
                    with st.expander(f"[{i+1}] {doc['metadata'].get('사업명', '제목 없음')}"):
                        st.write("📄 **메타데이터**")
                        st.json(doc["metadata"])
                        st.write("📝 **문서 내용**")
                        st.write(doc["content"])
            elif isinstance(docs, list) and len(docs) == 0:
                st.warning("검색된 문서가 없습니다.")
            else:
                st.info(docs.page_content)


        # 간단한 푸터
        st.markdown("---")
        st.caption("© 2026 Codeit-Part3-5Team. All rights reserved.")
    except Exception as e:
        st.error(f"[오류] 기능 실행 중 오류 발생: {str(e)}")
        # st.stop()  # 아래 코드 실행 안 함

if __name__ == "__main__":
    main_page()