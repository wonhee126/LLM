# movie_chatbot_app.py

import streamlit as st
import pandas as pd
import chromadb
import openai
import os
import sys
import time

# API 키 설정 (api_setting.py에서 임포트)
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    if current_script_dir not in sys.path:
        sys.path.append(current_script_dir)
        
    from api_setting import OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    print("OpenAI API 키 설정 완료!")
except ImportError:
    st.error("오류: 'api_setting.py'에서 'OPENAI_API_KEY'를 찾을 수 없습니다.")
    st.error("스크립트와 api_setting.py가 같은 폴더에 있는지 확인하고, api_setting.py에 OPENAI_API_KEY가 올바르게 설정되었는지 확인해주세요.")
    st.stop() # Streamlit 앱 중지
except Exception as e:
    st.error(f"API 키 설정 중 예기치 않은 오류 발생: {e}")
    st.stop() # Streamlit 앱 중지

# 설정 변수
CHROMA_DB_PERSIST_PATH = './chroma_db_movies'
CHROMA_COLLECTION_NAME = 'movie_embeddings_collection'

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" 
GPT_MODEL = "gpt-4o-mini" # 챗봇 답변 생성에 사용할 GPT 모델

# ChromaDB 로드 및 컬렉션 가져오기 (초기 로드)
@st.cache_resource # Streamlit 앱 실행 중 한 번만 로드하도록 캐싱
def load_chroma_collection():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_PATH)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        # st.success(f"ChromaDB 컬렉션 '{CHROMA_COLLECTION_NAME}' 로드 완료! (총 {collection.count()}개 아이템)") # 주석 처리
        return collection
    except Exception as e:
        st.error(f"오류: ChromaDB 컬렉션을 로드할 수 없습니다. '{CHROMA_DB_PERSIST_PATH}' 경로와 컬렉션 이름을 확인해주세요.")
        st.error("데이터가 ChromaDB에 로드되었는지 확인하려면 'load_movies_to_chromadb.py'를 먼저 실행해야 합니다.")
        st.stop() # Streamlit 앱 중지

# ChromaDB 컬렉션 로드
chroma_collection = load_chroma_collection()

# OpenAI 임베딩 함수 정의
def get_openai_embeddings(texts, model=OPENAI_EMBEDDING_MODEL, max_retries=3, delay=1):
    """
    OpenAI API를 사용하여 텍스트 리스트의 임베딩을 생성합니다.
    Rate Limit 등으로 인한 실패 시 재시도 로직 포함.
    """
    for i in range(max_retries):
        try:
            response = openai.embeddings.create(input=texts, model=model)
            return [d.embedding for d in response.data]
        except openai.APIStatusError as e:
            if e.status == 429:
                st.warning(f"OpenAI API Rate Limit 초과 (재시도 {i+1}/{max_retries}). {delay}초 대기 후 재시도...")
                time.sleep(delay)
                delay *= 2 
            else:
                st.error(f"OpenAI API 오류 발생: {e}")
                raise 
        except Exception as e:
            st.error(f"임베딩 생성 중 예기치 않은 오류 발생: {e}")
            raise 
    raise Exception(f"오류: OpenAI 임베딩 생성 실패 후 {max_retries}번 재시도.")

# ChromaDB 검색 함수
def search_similar_movies_in_chroma(user_query, collection, top_k=5):
    """
    사용자 쿼리를 임베딩하고 ChromaDB에서 유사한 영화를 검색합니다.
    """
    try:
        query_embedding = get_openai_embeddings([user_query], model=OPENAI_EMBEDDING_MODEL)[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,                     
            include=['metadatas', 'distances']   
        )
        
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            similar_movies = []
            for i in range(len(results['metadatas'][0])):
                movie_metadata = results['metadatas'][0][i]
                similar_movies.append(movie_metadata)
            return pd.DataFrame(similar_movies)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"영화 검색 중 오류 발생: {e}")
        return pd.DataFrame()

# GPT 답변 생성 함수
def generate_gpt_response(user_query, recommended_movies_df):
    """
    추천된 영화 정보를 바탕으로 GPT 모델에게 답변을 생성하도록 요청합니다.
    만약 추천 영화 정보가 없으면 GPT 모델의 일반 지식을 활용합니다.
    """
    if recommended_movies_df.empty:
        # ChromaDB에서 관련 영화를 찾지 못했을 경우 GPT에게 직접 질문합니다.
        print(f"DEBUG: ChromaDB에서 '{user_query}'에 대한 영화를 찾지 못했습니다. GPT에 직접 문의합니다.") # 디버깅용
        messages = [
            {"role": "system", "content": "당신은 사용자 질문에 답변하는 친절한 AI 어시스턴트입니다. 영화 관련 질문에 답변할 수 있습니다. 모르는 내용은 솔직하게 모른다고 말해주세요."},
            {"role": "user", "content": user_query} # 사용자 질문을 직접 GPT에게 전달
        ]
    else:
        # ChromaDB에서 관련 영화를 찾았을 경우, 해당 정보를 바탕으로 답변을 생성합니다.
        movie_info_strings = []
        for idx, row in recommended_movies_df.iterrows():
            info = f"제목: {row['title']}\n" \
                   f"장르: {row['genres']}\n" \
                   f"감독: {row['directors']}\n" \
                   f"주요 배우: {row['actors']}\n" \
                   f"줄거리: {row['overview']}\n" \
                   f"개봉일: {row['release_date']}\n" \
                   f"평점: {row['vote_average']}/10\n"
            movie_info_strings.append(info)
        
        prompt = fr"""사용자 질문: "{user_query}"

다음은 사용자 질문과 관련된 영화 정보입니다:

---
{'-'*50}
{"\n---\n".join(movie_info_strings)}
{'-'*50}

위 영화 정보를 바탕으로 사용자에게 친절하고 자연스럽게 답변하며 영화를 추천해주세요.
단, 다음과 같은 규칙을 지켜주세요:
1. 추천하는 영화의 '제목', '장르', '감독', '주요 배우', '줄거리','평점'을 간략하게 언급해주세요.
2. 친근하고 자연스러운 대화체로 답변해주세요.
3. 영화 정보를 과도하게 자세히 설명하기보다는 핵심적인 정보만 포함해주세요.
4. 사용자 질문에 직접적으로 답변하는 것처럼 보이도록 해주세요.
5. 영화 추천 외에 다른 질문에는 '저는 영화 추천에 특화된 챗봇입니다. 다른 질문은 도와드리기 어렵습니다.'라고 답변해주세요.
6. 추천하는 영화를 알려줄 때 글씨를 빼곡하게 나열만 하지말고 깔끔한 형식으로 알려주세요.
"""
        messages = [
            {"role": "system", "content": "당신은 영화 추천 전문가 챗봇입니다. 사용자의 질문과 제공된 영화 정보를 바탕으로 친절하고 자연스러운 영화 추천 답변을 제공합니다."},
            {"role": "user", "content": prompt}
        ]

    try:
        response = openai.chat.completions.create(
            model=GPT_MODEL, 
            messages=messages,
            max_tokens=500, 
            temperature=0.7 
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        st.error(f"OpenAI GPT API 오류 발생: {e}")
        return "죄송합니다. 영화 추천 서비스를 제공하는 데 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        st.error(f"알 수 없는 오류 발생: {e}")
        return "죄송합니다. 오류가 발생하여 요청을 처리할 수 없습니다."

# 질문 의도 분류 함수
def classify_query_intent(user_query, gpt_model=GPT_MODEL, max_retries=3, delay=1):
    """
    사용자 질문이 영화 추천 관련 질문인지 분류합니다.
    """
    classification_prompt = f"""다음 사용자 질문이 영화, 드라마, 또는 특정 작품 정보 요청과 관련 있는지 없는지 분류해주세요.
즉, **사용자가 영화나 드라마를 찾거나, 특정 영화/드라마에 대한 정보를 묻는 질문**이면 '영화추천'으로 분류하고,
**그 외의 질문 (예: 날씨, 일반 상식, 개인적인 질문 등)**이면 '기타질문'으로 분류합니다.
명확하게 '영화추천' 또는 '기타질문' 중 하나만 답변해주세요.

사용자 질문: "{user_query}"

분류:"""

    messages = [
        {"role": "system", "content": "사용자 질문의 의도를 '영화추천' 또는 '기타질문' 중 하나로만 분류하는 AI입니다. 영화 및 드라마 관련 질문은 모두 '영화추천'으로 분류합니다."}, 
        {"role": "user", "content": classification_prompt}
    ]

    for i in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=gpt_model,
                messages=messages,
                max_tokens=10, 
                temperature=0.0 
            )
            classification = response.choices[0].message.content.strip()
            print(f"질문 '{user_query}' 분류 결과: {classification}") 
            
            if "영화추천" in classification.lower(): 
                return "영화추천"
            elif "기타질문" in classification.lower(): 
                return "기타질문"
            else: 
                print(f"경고: GPT가 예상치 못한 분류 결과를 반환했습니다: '{classification}'. '기타질문'으로 처리합니다.")
                return "기타질문"
                
        except openai.APIError as e:
            if e.status == 429:
                st.warning(f"분류 API Rate Limit 초과 (재시도 {i+1}/{max_retries}). {delay}초 대기 후 재시도...")
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"OpenAI API (분류) 오류 발생: {e}")
                return "기타질문" 
        except Exception as e:
            st.error(f"질문 분류 중 예기치 않은 오류 발생: {e}")
            return "기타질문" 

# streamlit UI
st.set_page_config(page_title="영화 추천 챗봇", layout="centered")
st.title("🎬 영화 추천 챗봇")
st.markdown("궁금한 영화나 보고 싶은 영화 스타일을 알려주세요!")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! 어떤 영화를 찾으시나요? 저는 영화 추천에 특화된 챗봇입니다."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("보고 싶은 영화 스타일을 입력하세요..."):
    # 사용자 메시지 화면에 추가
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 챗봇 응답 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("질문 의도를 파악하고 영화 추천 정보를 탐색하고 있어요..."):
            # 질문 의도 분류
            intent = classify_query_intent(user_query, gpt_model=GPT_MODEL)
            
            if intent == '영화추천':
                recommended_movies_df = search_similar_movies_in_chroma(user_query, chroma_collection, top_k=3)
                gpt_response = generate_gpt_response(user_query, recommended_movies_df)
            else:
                gpt_response = "저는 영화 추천에 특화된 챗봇입니다. 다른 질문은 도와드리기 어렵습니다."
            
            st.markdown(gpt_response)
            st.session_state.messages.append({"role": "assistant", "content": gpt_response})