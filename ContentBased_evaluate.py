import random
import numpy as np
import pandas as pd
from ast import literal_eval
from ContentBased_Modeling import ListingRecommender

class RecommenderEvaluator:
    def __init__(self, listing_recommender):
        self.listing_recommender = listing_recommender
        self.listing = listing_recommender.listing
        # 초기화 시 user-listing 매핑 생성
        self.user_listings_map = self._create_user_listings_map()
        
    def _create_user_listings_map(self):
        """
        유저별 방문한 listing_id 매핑을 미리 생성
        """
        user_listings = {}
        for _, row in self.listing.iterrows():
            visitors = row['visitors']
            if isinstance(visitors, list):
                for user in visitors:
                    if user not in user_listings:
                        user_listings[user] = []
                    user_listings[user].append(row['listing_id'])
        return user_listings
    
    def precision_at_k(self, recommended_ids, relevant_items, k):
        """Precision@k 계산"""
        if not recommended_ids or not relevant_items:
            return 0.0
        recommended_at_k = set(recommended_ids[:k])
        relevant_set = set(relevant_items)
        hits = len(recommended_at_k & relevant_set)
        return hits / k if k > 0 else 0.0
    
    def recall_at_k(self, recommended_ids, relevant_items, k):
        """Recall@k 계산"""
        if not recommended_ids or not relevant_items:
            return 0.0
        recommended_at_k = set(recommended_ids[:k])
        relevant_set = set(relevant_items)
        hits = len(recommended_at_k & relevant_set)
        return hits / len(relevant_set)
    
    def get_user_visited_listings(self, user_id):
        """캐시된 매핑에서 유저의 방문 기록 조회"""
        return self.user_listings_map.get(user_id, [])
    
    def evaluate_model(self, sample_size=100, k=10):
        """최적화된 평가 함수"""
        # 유효한 유저만 필터링 (2개 이상 방문 기록이 있는 유저)
        valid_users = [
            user for user, listings in self.user_listings_map.items()
            if len(listings) >= 2
        ]
        
        if not valid_users:
            raise ValueError("No valid users found for evaluation")
        
        # 샘플 크기 조정
        sample_size = min(sample_size, len(valid_users))
        sampled_users = random.sample(valid_users, sample_size)
        
        results = []
        print(f"\nEvaluating {sample_size} users...")
        
        for i, user_id in enumerate(sampled_users, 1):
            try:
                # 방문 기록 가져오기
                user_visited_listings = self.user_listings_map[user_id]
                
                # 방문 기록 분할 (학습용/평가용)
                train_size = max(1, len(user_visited_listings) // 2)
                train_listings = user_visited_listings[:train_size]
                test_listings = user_visited_listings[train_size:]
                
                if not train_listings or not test_listings:
                    continue
                
                # 추천 생성
                _, recommended_ids = self.listing_recommender.get_recommendations_with_user_preference(
                    train_listings,
                    topn=k
                )
                
                # 성능 평가
                precision = self.precision_at_k(recommended_ids, test_listings, k)
                recall = self.recall_at_k(recommended_ids, test_listings, k)
                
                results.append({
                    'user_id': user_id,
                    'total_visits': len(user_visited_listings),
                    'precision': precision,
                    'recall': recall
                })
                
                if i % 5 == 0:
                    print(f"Processed {i}/{sample_size} users...")
                    
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                continue
        
        # 결과 계산
        if not results:
            return 0, 0, pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        avg_precision = results_df['precision'].mean()
        avg_recall = results_df['recall'].mean()
        
        print("\n=== Evaluation Results ===")
        print(f"Users evaluated: {len(results_df)}")
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Average Recall@{k}: {avg_recall:.4f}")
        
        return avg_precision, avg_recall, results_df