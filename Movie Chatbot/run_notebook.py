# run_notebook.py

# import sys
# import os
import pandas as pd
# import requests
# import json
# import time

from api_setting import TMDB_API_KEY, BASE_URL
from data import fetch_movie_data

print("모듈 임포트 및 경로 설정 완료!")
print(f"API 키 테스트를 위한 BASE_URL: {BASE_URL}")

# 데이터 정제 및 가공 함수 정의
def process_movie_data(df_raw):
    """
    수집된 영화 데이터를 정제하고 임베딩을 위한 텍스트로 가공합니다.
    """
    df_cleaned = df_raw[[
        'id', 'title', 'original_title', 'overview', 'release_date',
        'genres', 'directors', 'actors', 'vote_average', 'popularity', 'runtime', 'tagline'
    ]].copy()

    df_cleaned.fillna('', inplace=True)
    df_cleaned['overview'].replace('', '줄거리 정보 없음', inplace=True)
    df_cleaned['tagline'].replace('', '태그라인 정보 없음', inplace=True)

    df_cleaned['genres_str'] = df_cleaned['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    df_cleaned['directors_str'] = df_cleaned['directors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    df_cleaned['actors_str'] = df_cleaned['actors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')

    df_cleaned['embedding_text'] = df_cleaned.apply(
        lambda row: f"제목: {row['title']}. 원제: {row['original_title']}. "
                    f"장르: {row['genres_str']}. "
                    f"줄거리: {row['overview']}. "
                    f"감독: {row['directors_str']}. "
                    f"배우: {row['actors_str']}. "
                    f"개봉일: {row['release_date']}. "
                    f"평점: {row['vote_average']}. "
                    f"인기: {row['popularity']}. "
                    f"상영시간: {row['runtime']}분. "
                    f"태그라인: {row['tagline']}.", axis=1
    )
    return df_cleaned

# 데이터 수집 및 처리 파이프라인 실행
if __name__ == "__main__":
    print("--- 영화 데이터 수집 및 처리 파이프라인 시작 ---")
    
    # 데이터 수집 (원하는 페이지 수로 설정)
    num_pages_to_fetch = 5 # 테스트 시에는 적게, 실제 수집 시에는 더 많이 설정
    
    print("\n--- 1. 영화 데이터 수집 시작 ---")
    # data.py에서 임포트한 fetch_movie_data 함수를 사용합니다.
    # TMDB_API_KEY는 api_setting.py에서 임포트되어 이미 사용 가능합니다.
    df_movies_raw = fetch_movie_data(TMDB_API_KEY, num_pages=num_pages_to_fetch)
    print("--- 영화 데이터 수집 완료 ---")

    if not df_movies_raw.empty:
        print("\n--- 2. 데이터 정제 및 가공 시작 ---")
        df_movies_processed = process_movie_data(df_movies_raw)
        print("--- 데이터 정제 및 가공 완료 ---")

        # 데이터 저장
        output_csv_path = 'tmdb_movies_processed_data.csv'
        df_movies_processed.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n--- 정제된 영화 데이터가 '{output_csv_path}' 파일로 저장되었습니다. ---")

        print("\n최종 처리된 데이터의 첫 5줄:")
        print(df_movies_processed[['id', 'title', 'genres_str', 'directors_str', 'actors_str', 'embedding_text']].head())
        print("\n최종 처리된 데이터 정보:")
        print(df_movies_processed.info())
    else:
        print("수집된 영화 데이터가 없어 추가 처리를 진행하지 않습니다.")

    print("\n--- 영화 데이터 파이프라인 종료 ---")