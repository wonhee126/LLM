# data.py

# import os
# import sys

import requests
import pandas as pd
import time
from api_setting import TMDB_API_KEY, BASE_URL

def fetch_movie_data(api_key, num_pages=100, language="ko-KR"):
    """
    TMDB API에서 지정된 페이지 수만큼 인기 영화 데이터를 가져옵니다.
    각 영화에 대해 추가 상세 정보를 가져와 통합합니다.
    """
    all_movies = []
    
    print(f"총 {num_pages} 페이지의 영화 데이터를 수집합니다...")

    for page in range(1, num_pages + 1):
        # 인기 영화 목록 가져오기
        popular_movies_endpoint = f"{BASE_URL}/movie/popular"
        popular_params = {
            "api_key": api_key,
            "language": language,
            "page": page
        }
        popular_response = requests.get(popular_movies_endpoint, params=popular_params)

        if popular_response.status_code == 200:
            popular_data = popular_response.json()
            movies_on_page = popular_data.get('results', [])

            if not movies_on_page:
                print(f"페이지 {page}에서 더 이상 영화를 찾을 수 없습니다. 수집을 중단합니다.")
                break

            print(f"페이지 {page}/{num_pages} 처리 중... (영화 {len(movies_on_page)}개)")

            for movie_summary in movies_on_page:
                movie_id = movie_summary.get('id')
                if not movie_id:
                    continue

                # 각 영화의 상세 정보 가져오기 (감독, 배우 등)
                movie_detail_endpoint = f"{BASE_URL}/movie/{movie_id}"
                detail_params = {
                    "api_key": api_key,
                    "language": language,
                    "append_to_response": "credits" # 출연진(cast) 및 제작진(crew) 정보 포함
                }
                detail_response = requests.get(movie_detail_endpoint, params=detail_params)

                if detail_response.status_code == 200:
                    movie_detail = detail_response.json()
                    combined_movie_info = {
                        "id": movie_detail.get('id'),
                        "title": movie_detail.get('title'),
                        "original_title": movie_detail.get('original_title'),
                        "overview": movie_detail.get('overview'),
                        "release_date": movie_detail.get('release_date'),
                        "genres": [g['name'] for g in movie_detail.get('genres', [])],
                        "vote_average": movie_detail.get('vote_average'),
                        "popularity": movie_detail.get('popularity'),
                        "poster_path": movie_detail.get('poster_path'),
                        "backdrop_path": movie_detail.get('backdrop_path'),
                        "tagline": movie_detail.get('tagline'),
                        "runtime": movie_detail.get('runtime'), # 상영 시간 (분)
                    }

                    directors = [
                        crew['name'] for crew in movie_detail.get('credits', {}).get('crew', [])
                        if crew.get('job') == 'Director'
                    ]
                    combined_movie_info['directors'] = directors

                    actors = [
                        cast['name'] for cast in movie_detail.get('credits', {}).get('cast', [])[:5] # 상위 5명
                    ]
                    combined_movie_info['actors'] = actors

                    all_movies.append(combined_movie_info)
                else:
                    print(f"영화 ID {movie_id} 상세 정보 가져오기 실패: {detail_response.status_code}")
                
                time.sleep(0.05) # 50ms 대기

            time.sleep(0.5) # 페이지당 0.5초 대기
        else:
            print(f"페이지 {page} 인기 영화 목록 가져오기 실패: {popular_response.status_code}")
            print(f"오류 메시지: {popular_response.text}")
            break # 오류 발생 시 중단

    print(f"총 {len(all_movies)}개의 영화 데이터를 수집했습니다.")
    return pd.DataFrame(all_movies)

# 이 파일이 직접 실행될 때만 실행되는 코드 블록
if __name__ == "__main__":
    # 데이터 수집 함수 실행 예시
    num_pages_to_fetch = 10 # 테스트를 위해 페이지 수를 줄여보세요
    df_movies_raw = fetch_movie_data(TMDB_API_KEY, num_pages=num_pages_to_fetch)
    
    print("\n수집된 원본 영화 데이터의 첫 5줄:")
    print(df_movies_raw.head())
    print("\n수집된 원본 영화 데이터 정보:")
    print(df_movies_raw.info())

