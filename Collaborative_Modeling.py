import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class Modeling:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.data['visitors'] = self.data['visitors'].apply(eval)
        self.data['user_ratings'] = self.data['user_ratings'].apply(eval)
        self.data = self.data.explode(['visitors', 'user_ratings']).rename(
            columns={'visitors': 'user_id', 'user_ratings': 'user_rating'}
        )
        self.sampled_data = self.data.sample(frac=0.2, random_state=42)
        self.listing_similarity = None
        self.listing_indices = []

    def create_similarity_matrix(self):
        self.sampled_data['interaction_str'] = self.sampled_data.apply(
            lambda x: f"{x['user_id']}_{x['user_rating']}", axis=1
        )
        listing_interactions = self.sampled_data.groupby('listing_id')['interaction_str'].apply(lambda x: ' '.join(x))
        count_vectorizer = CountVectorizer()
        interaction_matrix = count_vectorizer.fit_transform(listing_interactions)
        self.listing_similarity = cosine_similarity(interaction_matrix, interaction_matrix)
        self.listing_indices = listing_interactions.index.tolist()

    def get_similar_listings(self, listing_id, top_k=5):
        if listing_id not in self.listing_indices:
            return []
        idx = self.listing_indices.index(listing_id)
        sim_scores = sorted(enumerate(self.listing_similarity[idx]), key=lambda x: x[1], reverse=True)[1:top_k + 1]
        similar_listing_indices = [self.listing_indices[i[0]] for i in sim_scores]
        return similar_listing_indices