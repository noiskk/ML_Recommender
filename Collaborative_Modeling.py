import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

class ItemBasedModeling:
    def __init__(self, data_path):
        # 데이터 로드 및 전처리
        self.data = pd.read_csv(data_path)
        self.data['visitors'] = self.data['visitors'].apply(literal_eval)
        
        # 방문자 데이터를 행렬 형태로 변환
        self.listing_user_matrix = self._create_listing_user_matrix()
        self.listing_similarity = None
        self.listing_indices = None

    def _create_listing_user_matrix(self):
        """listing-user 상호작용 행렬 생성"""
        listings = []
        users = set()
        
        # 모든 방문자 수집
        for _, row in self.data.iterrows():
            listings.append(row['listing_id'])
            users.update(row['visitors'])
        
        # 행렬 생성
        matrix = pd.DataFrame(0, index=listings, columns=list(users))
        
        # 방문 데이터 채우기
        for _, row in self.data.iterrows():
            for visitor in row['visitors']:
                matrix.loc[row['listing_id'], visitor] = 1
                
        return matrix

    def create_similarity_matrix(self):
        """숙소 간 유사도 행렬 계산"""
        self.listing_similarity = cosine_similarity(self.listing_user_matrix)
        self.listing_indices = self.listing_user_matrix.index.tolist()

    def get_similar_listings(self, listing_id, top_k=5):
        """특정 숙소와 유사한 숙소 추천"""
        if listing_id not in self.listing_indices:
            return []
            
        idx = self.listing_indices.index(listing_id)
        sim_scores = list(enumerate(self.listing_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]  # 자기 자신 제외
        
        similar_listings = [self.listing_indices[i[0]] for i in sim_scores]
        return similar_listings