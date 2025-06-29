# load_movies_to_chromadb.py

import pandas as pd
import chromadb
import openai
import os
import sys
import time
from api_setting import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY
print("OpenAI API 키 설정 완료!")


# 설정 변수 ---
CSV_FILE_PATH = 'tmdb_movies_processed_data.csv'
CHROMA_DB_PERSIST_PATH = './chroma_db_movies' # ChromaDB 데이터가 저장될 로컬 폴더 경로
CHROMA_COLLECTION_NAME = 'movie_embeddings_collection' # ChromaDB 컬렉션 이름

# 사용할 OpenAI 임베딩 모델
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" 

# 대용량 파일 처리 및 OpenAI API Rate Limit을 고려한 청크 크기
# 한 번에 OpenAI API에 보낼 텍스트의 개수 (영화 수)
OPENAI_EMBEDDING_CHUNK_SIZE = 500 

# CSV 파일을 읽을 때의 청크 크기 (pandas read_csv용)
# 보통 OpenAI 임베딩 청크 사이즈보다 크거나 같게 설정하여 IO 부담 줄이기
PANDAS_READ_CHUNK_SIZE = 1000 

# 파일 존재 여부 확인
if not os.path.exists(CSV_FILE_PATH):
    print(f"오류: '{CSV_FILE_PATH}' 파일을 찾을 수 없습니다.")
    print("먼저 데이터 수집 스크립트(run_notebook.py)를 실행하여 이 파일을 생성해주세요.")
    sys.exit(1)

print(f"'{CSV_FILE_PATH}' 파일을 읽어 OpenAI 임베딩을 사용하여 ChromaDB에 저장합니다.")
print(f"  사용할 OpenAI 임베딩 모델: {OPENAI_EMBEDDING_MODEL}")
print(f"  ChromaDB 저장 경로: {CHROMA_DB_PERSIST_PATH}")
print(f"  ChromaDB 컬렉션 이름: {CHROMA_COLLECTION_NAME}")
print(f"  OpenAI 임베딩 청크 크기 (영화 수): {OPENAI_EMBEDDING_CHUNK_SIZE}")
print(f"  Pandas CSV 읽기 청크 크기 (영화 수): {PANDAS_READ_CHUNK_SIZE}")

# OpenAI 임베딩 함수 정의
def get_openai_embeddings(texts, model=OPENAI_EMBEDDING_MODEL, max_retries=5, delay=1):
    """
    OpenAI API를 사용하여 텍스트 리스트의 임베딩을 생성합니다.
    Rate Limit 등으로 인한 실패 시 재시도 로직 포함.
    """
    for i in range(max_retries):
        try:
            response = openai.embeddings.create(input=texts, model=model)
            # response.data는 임베딩 객체 리스트를 반환, 각 객체의 .embedding 속성에 실제 벡터가 있음
            embeddings = [d.embedding for d in response.data] 
            return embeddings
        except openai.APIStatusError as e:
            if e.status == 429: # Too Many Requests (Rate Limit Error)
                print(f"경고: OpenAI API Rate Limit 초과 (재시도 {i+1}/{max_retries}). {delay}초 대기 후 재시도...")
                time.sleep(delay)
                delay *= 2 # 지연 시간 증가
            else:
                print(f"오류: OpenAI API Status Error - {e}")
                raise # 다른 API 오류는 즉시 발생
        except Exception as e:
            print(f"오류: 임베딩 생성 중 예기치 않은 오류 발생: {e}")
            raise # 다른 오류는 즉시 발생
    raise Exception(f"오류: OpenAI 임베딩 생성 실패 후 {max_retries}번 재시도.")

# ChromaDB 클라이언트 및 컬렉션 설정
try:
    # ChromaDB 클라이언트 초기화 (데이터를 지정된 경로에 영구 저장)
    client = chromadb.PersistentClient(path=CHROMA_DB_PERSIST_PATH)
    
    # 기존 컬렉션이 있다면 삭제하고 새로 생성 (초기화 목적)
    # 데이터를 매번 새로 로드할 때 유용합니다.
    # 운영 환경에서는 이 라인을 주석 처리하여 데이터가 보존되도록 해야 합니다.
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"기존 컬렉션 '{CHROMA_COLLECTION_NAME}'을(를) 삭제했습니다.")
    except Exception as e:
        print(f"컬렉션 '{CHROMA_COLLECTION_NAME}'이(가) 존재하지 않거나 삭제할 필요가 없습니다. (info: {e})")

    # ChromaDB 컬렉션 생성
    # metadata={"hnsw:space": "cosine"}은 벡터 유사도 계산 시 코사인 유사도를 사용하도록 설정
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} 
    )
    print(f"ChromaDB 컬렉션 '{CHROMA_COLLECTION_NAME}' 생성 완료!")
    
except Exception as e:
    print(f"ChromaDB 설정 중 오류 발생: {e}")
    sys.exit(1)

# 청크 단위로 데이터 읽고 임베딩 및 ChromaDB에 저장
processed_rows_count = 0
pandas_chunk_number = 0
openai_batch_count = 0
total_rows_in_csv = 0

# CSV 파일의 총 행 수 미리 계산 (진행률 표시용, 헤더 제외)
try:
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
        total_rows_in_csv = sum(1 for row in f) - 1 
    print(f"CSV 파일 총 영화 수 (헤더 제외): {total_rows_in_csv}개")
except Exception as e:
    print(f"오류: CSV 파일의 총 행 수를 계산할 수 없습니다: {e}")
    total_rows_in_csv = "알 수 없음"

# 임베딩할 텍스트, 메타데이터, ID를 임시로 담을 리스트
batch_texts = []
batch_metadatas = []
batch_ids = []

try:
    # Pandas를 사용하여 CSV 파일을 청크 단위로 읽기
    # chunksize는 pandas가 한 번에 메모리에 로드할 행의 수입니다.
    for pandas_chunk_df in pd.read_csv(CSV_FILE_PATH, chunksize=PANDAS_READ_CHUNK_SIZE):
        pandas_chunk_number += 1
        print(f"\n[Pandas 청크 {pandas_chunk_number} 처리 중...]")

        if 'embedding_text' not in pandas_chunk_df.columns:
            print(f"경고: Pandas 청크 {pandas_chunk_number}에 'embedding_text' 컬럼이 없습니다. 이 청크를 건너뜁니다.")
            continue
        
        # 현재 Pandas 청크의 각 행을 순회하며 데이터 준비
        for idx, row in pandas_chunk_df.iterrows():
            embedding_text = str(row['embedding_text']).strip() # 혹시 모를 숫자/None값 대비
            
            # 유효한 임베딩 텍스트가 아니면 건너뛰기
            if pd.isna(embedding_text) or not embedding_text: 
                #print(f"  - 경고: ID {row['id']} 영화의 'embedding_text'가 비어있어 건너뜁니다.")
                continue
            
            # ChromaDB에 저장할 메타데이터 (원하는 컬럼 모두 포함)
            metadata = {
                "movie_id": str(row['id']), 
                "title": str(row['title']),
                "original_title": str(row['original_title']),
                "genres": str(row['genres_str']),
                "directors": str(row['directors_str']),
                "actors": str(row['actors_str']),
                "release_date": str(row['release_date']),
                "vote_average": float(row['vote_average']),
                "popularity": float(row['popularity']),
                "runtime": float(row['runtime']),
                "tagline": str(row['tagline']),
                "overview": str(row['overview']),
            }
            
            # ChromaDB에 추가할 배치 리스트에 데이터 추가
            batch_texts.append(embedding_text)
            batch_metadatas.append(metadata)
            batch_ids.append(f"movie_{row['id']}") # 각 영화의 고유 ID

            # OpenAI 임베딩 청크 크기에 도달하면 API 호출 및 ChromaDB에 추가
            if len(batch_texts) >= OPENAI_EMBEDDING_CHUNK_SIZE:
                openai_batch_count += 1
                print(f"  - OpenAI 임베딩 배치 {openai_batch_count} 처리 중 ({len(batch_texts)}개 영화)...")
                
                chunk_embeddings = get_openai_embeddings(batch_texts, model=OPENAI_EMBEDDING_MODEL)
                
                collection.add(
                    embeddings=chunk_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts, # 원본 텍스트도 함께 저장
                    ids=batch_ids
                )
                processed_rows_count += len(batch_texts)
                print(f"  - {len(batch_texts)}개 영화 데이터를 ChromaDB에 추가 완료. 총 {processed_rows_count}개 처리됨.")
                
                # 배치 초기화
                batch_texts = []
                batch_metadatas = []
                batch_ids = []

    # 마지막으로 남은 데이터 처리 (청크 사이즈 미만일 수 있음)
    if batch_texts:
        openai_batch_count += 1
        print(f"\n[최종 배치 {openai_batch_count} 처리 중 ({len(batch_texts)}개 영화)...]")
        final_chunk_embeddings = get_openai_embeddings(batch_texts, model=OPENAI_EMBEDDING_MODEL)
        
        collection.add(
            embeddings=final_chunk_embeddings,
            metadatas=batch_metadatas,
            documents=batch_texts,
            ids=batch_ids
        )
        processed_rows_count += len(batch_texts)
        print(f"  - 최종 {len(batch_texts)}개 영화 데이터를 ChromaDB에 추가 완료. 총 {processed_rows_count}개 처리됨.")

    print(f"\n--- 모든 영화 데이터의 ChromaDB 저장 완료! ---")
    print(f"총 {processed_rows_count}개 영화의 임베딩이 '{CHROMA_COLLECTION_NAME}' 컬렉션에 저장되었습니다.")

    # 저장된 데이터 수 최종 확인
    print(f"ChromaDB 컬렉션 '{CHROMA_COLLECTION_NAME}'의 최종 아이템 수: {collection.count()}개")

except pd.errors.EmptyDataError:
    print(f"오류: '{CSV_FILE_PATH}' 파일이 비어 있습니다. 데이터를 먼저 수집해주세요.")
    sys.exit(1)
except Exception as e:
    print(f"데이터 처리 또는 ChromaDB 저장 중 치명적인 오류 발생: {e}")
    sys.exit(1)
