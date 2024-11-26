import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


from Collaborative_Modeling import Modeling

class ModelEvaluator:
    def __init__(self, modeling, k=5):
        self.modeling = modeling
        self.k = k

    def precision_at_k_for_user(self, user_id):
        user_data = self.modeling.sampled_data[
            (self.modeling.sampled_data['user_id'] == user_id) & 
            (self.modeling.sampled_data['user_rating'] >= 90)
        ]['listing_id'].unique()

        if len(user_data) == 0:
            return 0
        
        sample_listing_id = user_data[0]
        recommended_ids = self.modeling.get_similar_listings(sample_listing_id, top_k=self.k)
        relevant_at_k = set(recommended_ids) & set(user_data)
        return len(relevant_at_k) / self.k

    def mean_precision_at_k_all_users(self):
        unique_users = self.modeling.sampled_data['user_id'].unique()
        precision_scores = [
            self.precision_at_k_for_user(user_id) for user_id in unique_users
        ]
        return np.mean(precision_scores) if precision_scores else 0

    def calculate_mae_rmse(self):
        actual_ratings = self.modeling.sampled_data['user_rating'].values
        predicted_ratings = np.random.randint(1, 101, len(actual_ratings))  # Placeholder for predicted ratings
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        return mae, rmse

    def evaluate(self):
        self.modeling.create_similarity_matrix()
        mean_precision_score = self.mean_precision_at_k_all_users()
        mae, rmse = self.calculate_mae_rmse()
        return mean_precision_score, mae, rmse

    def print_results(self, mean_precision_score, mae, rmse):
        print("Evaluation completed.")
        print(f"Mean Precision@{self.k}: {mean_precision_score}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")