import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import time

class ItemBasedRecommender:
    def __init__(self, filepath):
        print("데이터 로딩 중...")
        self.listing = pd.read_csv(filepath)
        self.listing['visitors'] = self.listing['visitors'].apply(literal_eval)
        print(f"데이터 로드 완료: {len(self.listing)} 개의 리스팅")
        
        print("사용자-아이템 행렬 생성 중...")
        self.user_item_matrix = self._create_sparse_user_item_matrix()
        print("사용자-아이템 행렬 생성 완료")
        
        # 초기화 시점에는 유사도 행렬을 계산하지 않음
        self.item_similarity_matrix = None
    
    def _create_sparse_user_item_matrix(self):
        """사용자-아이템 희소 행렬 생성"""
        print("- 고유 사용자와 리스팅 식별 중...")
        # 모든 고유 사용자와 리스팅 식별
        unique_users = set()
        unique_listings = set(self.listing['listing_id'])
        
        for visitors in self.listing['visitors']:
            unique_users.update(visitors)
        
        print(f"- 식별된 고유 사용자 수: {len(unique_users)}")
        print(f"- 식별된 고유 리스팅 수: {len(unique_listings)}")
        
        # 사용자와 리스팅에 대한 고유 인덱스 매핑
        print("- 인덱스 매핑 생성 중...")
        user_to_idx = {user: idx for idx, user in enumerate(sorted(unique_users))}
        listing_to_idx = {listing: idx for idx, listing in enumerate(sorted(unique_listings))}
        
        # 데이터 준비
        print("- 희소 행렬 데이터 준비 중...")
        rows, cols = [], []
        for _, row in self.listing.iterrows():
            listing_idx = listing_to_idx[row['listing_id']]
            for user in row['visitors']:
                rows.append(user_to_idx[user])
                cols.append(listing_idx)
        
        # 희소 행렬 생성
        print("- 최종 희소 행렬 생성 중...")
        sparse_matrix = csr_matrix(
            (np.ones(len(rows)), (rows, cols)), 
            shape=(len(user_to_idx), len(listing_to_idx))
        )
        
        # 역 매핑 저장
        self.idx_to_user = {v: k for k, v in user_to_idx.items()}
        self.idx_to_listing = {v: k for k, v in listing_to_idx.items()}
        
        print(f"- 생성된 행렬 크기: {sparse_matrix.shape}")
        return sparse_matrix
    
    def create_similarity_matrix(self, batch_size=1000):
        """아이템 간 유사도 행렬 계산"""
        print("유사도 행렬 계산 시작...")
        start_time = time.time()
        
        print("- 행렬 변환 중...")
        item_matrix = self.user_item_matrix.T.toarray()
        n_items = item_matrix.shape[0]
        
        print(f"- 전체 아이템 수: {n_items}")
        print(f"- 배치 크기: {batch_size}")
        
        # 결과를 저장할 빈 행렬 초기화
        self.item_similarity_matrix = np.zeros((n_items, n_items))
        
        # 배치 단위로 유사도 계산
        for i in range(0, n_items, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, n_items)
            
            print(f"- 배치 처리 중: {batch_start+1}~{batch_end}/{n_items} 아이템")
            batch_similarities = cosine_similarity(
                item_matrix[batch_start:batch_end],
                item_matrix
            )
            
            # 계산된 유사도를 결과 행렬에 할당
            self.item_similarity_matrix[batch_start:batch_end] = batch_similarities
            
            # 진행률 출력
            progress = (batch_end / n_items) * 100
            elapsed = time.time() - start_time
            print(f"  진행률: {progress:.1f}% (경과 시간: {elapsed:.1f}초)")

            print(f"유사도 행렬 계산 완료 (크기: {self.item_similarity_matrix.shape}, 총 소요시간: {time.time() - start_time:.2f}초)")

    def get_user_visited_listings(self, user_id):
        """
        특정 사용자가 방문한 리스팅 목록을 반환
        
        Args:
            user_id (int): 사용자 ID
        
        Returns:
            list: 방문한 리스팅 ID 리스트
        """
        visited_listings = []
        for index, row in self.listing.iterrows():
            if user_id in row['visitors']:
                visited_listings.append(row['listing_id'])
        return visited_listings
        
    def get_item_based_recommendations(self, user_id, topn=10):
        """
        특정 리스팅과 유사한 다른 리스팅 추천
        
        Args:
            listing_id (int): 기준이 되는 리스팅 ID
            topn (int): 추천할 리스팅 개수
        
        Returns:
            tuple: (추천된 리스팅 데이터프레임, 추천된 리스팅 ID 리스트)
        """

        visited_listings = self.get_user_visited_listings(user_id)
        listing_id = visited_listings[0]

        # 주어진 리스팅의 인덱스 찾기
        try:
            listing_index = list(self.listing['listing_id']).index(listing_id)
        except ValueError:
            raise ValueError(f"Listing ID {listing_id}를 찾을 수 없습니다.")
        
        # 해당 리스팅과 다른 리스팅들 간의 유사도
        similar_items = self.item_similarity_matrix[listing_index]
        
        # 자기 자신 제외 및 유사도 기준 정렬
        similar_indices = similar_items.argsort()[::-1][1:topn+1]
        
        # 추천 리스팅 선택
        recommended_listings = self.listing.iloc[similar_indices]
        recommended_ids = recommended_listings['listing_id'].tolist()
        
        return recommended_listings, recommended_ids
    

    