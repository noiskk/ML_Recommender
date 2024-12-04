import random
import numpy as np
import pandas as pd
from ast import literal_eval
from ContentBased_Modeling import ContentBasedModeling

class ContentBasedEvaluator:
    def __init__(self, model, k=10, test_file=None):
        self.model = model
        self.k = k
        if test_file:
            self.test_data = pd.read_csv(test_file)
            self.test_data['visitors'] = self.test_data['visitors'].apply(literal_eval)
        self.user_listings_map = self._create_user_listings_map()
    
    def _create_user_listings_map(self):
        """유저별 방문한 listing_id 매핑 생성"""
        user_listings = {}
        for _, row in self.test_data.iterrows():
            visitors = row['visitors']
            listing_id = row['listing_id']
            for user in visitors:
                if user not in user_listings:
                    user_listings[user] = []
                user_listings[user].append(listing_id)
        return user_listings
    
    def evaluate_single_user(self, user_id, k=10):
        """단일 사용자에 대한 추천 성능 평가"""
        # 사용자의 전체 방문 기록 가져오기
        user_listings = self.user_listings_map.get(user_id, [])
        
        if len(user_listings) < 2:  # 최소 2개 이상의 방문 기록 필요
            return None
        
        # 방문 기록을 train/test로 분할
        train_size = len(user_listings) // 2
        train_listings = user_listings[:train_size]
        test_listings = user_listings[train_size:]
        
        try:
            # 추천 생성
            _, recommended_ids = self.model.get_recommendations_with_user_preference(
                train_listings, topn=k
            )
            
            # Precision@K 계산
            hits = len(set(recommended_ids) & set(test_listings))
            precision = hits / k if k > 0 else 0
            
            # Recall 계산
            recall = hits / len(test_listings) if test_listings else 0
            
            return {
                'user_id': user_id,
                'precision': precision,
                'recall': recall,
                'num_train': len(train_listings),
                'num_test': len(test_listings),
                'num_recommendations': len(recommended_ids)
            }
        except Exception as e:
            print(f"Error evaluating user {user_id}: {str(e)}")
            return None
    
    def evaluate_model(self, sample_size=100):
        """전체 모델 성능 평가"""
        print(f"\nEvaluating model on {sample_size} users...")
        
        # 유효한 사용자 선택 (2개 이상의 방문 기록이 있는 사용자)
        valid_users = [
            user for user, listings in self.user_listings_map.items()
            if len(listings) >= 2
        ]
        
        if not valid_users:
            raise ValueError("No valid users found for evaluation")
        
        # 샘플 크기 조정
        sample_size = min(sample_size, len(valid_users))
        sampled_users = random.sample(valid_users, sample_size)
        
        # 평가 실행
        results = []
        for i, user_id in enumerate(sampled_users, 1):
            result = self.evaluate_single_user(user_id, k=self.k)
            if result:
                results.append(result)
            
            if i % 10 == 0:
                print(f"Processed {i}/{sample_size} users...")
        
        # 결과 분석
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        print("\n=== Evaluation Results ===")
        print(f"Number of users evaluated: {len(results_df)}")
        print("\nMetrics Summary:")
        print(f"Average Precision@{self.k}: {results_df['precision'].mean():.4f}")
        print(f"Average Recall@{self.k}: {results_df['recall'].mean():.4f}")
        print("\nDetailed Statistics:")
        print(results_df[['precision', 'recall']].describe())
        
        return results_df