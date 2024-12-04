import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval

class ContentBasedModeling:
    def __init__(self, filepath):
        # 데이터 로드
        listing = pd.read_csv(filepath)
        listing['visitors'] = listing['visitors'].apply(literal_eval)
        
        # 필요한 열만 추출
        amenities_columns = [
            "TV", "Internet", "Shampoo", "Suitable for Events", 
            "Washer / Dryer", "Pool", "Hair Dryer", "Smoke Detector", "Cable TV", 
            "Laptop Friendly Workspace", "Cat(s)", "Doorman", "Washer", "Heating", 
            "Breakfast", "Safety Card", "Hot Tub", "Pets live on this property", 
            "Free Parking on Premises", "Dryer", "Essentials", "Iron", "Wireless Internet", 
            "Dog(s)", "Pets Allowed", "Buzzer/Wireless Intercom", "Gym", "24-Hour Check-in", 
            "Fire Extinguisher", "Hangers", "Elevator in Building", "Other pet(s)", 
            "Lock on Bedroom Door", "Wheelchair Accessible", "Indoor Fireplace", 
            "Smoking Allowed", "Kitchen", "First Aid Kit", "Air Conditioning", 
            "Family/Kid Friendly", "Carbon Monoxide Detector"
        ]
        
        # 편의 시설 데이터를 이진화 (True/False를 1/0으로 변환)
        for col in amenities_columns:
            listing[col] = listing[col].astype(int)

        # 유사도 계산에 사용될 features
        self.df_relevant = listing[['listing_id', 'property_type', 'room_type', 'accommodates', 
                                    'bedrooms', 'price', 'city', 'visitors'] + amenities_columns]
        
        # 원-핫 인코딩 처리
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(self.df_relevant[['property_type', 'room_type', 'city']])
        
        # 원-핫 인코딩된 특성 데이터프레임에 추가
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['property_type', 'room_type', 'city']))
        self.encoded_df = pd.concat([self.df_relevant.reset_index(drop=True), encoded_df], axis=1)
        self.encoded_df = self.encoded_df.drop(columns=['property_type', 'room_type', 'city'])
        
        # price 형변환
        self.encoded_df['price'] = self.encoded_df['price'].replace({'\\$': ''}, regex=True).astype(float)
        
        # 필요한 특성만 선택 (편의 시설 포함)
        self.features = self.encoded_df.drop(columns=['listing_id', 'visitors'])
        self.listing = listing
        self.feature_matrix = self.features.values  # Feature matrix for cosine similarity calculation
        
    def generate_user_profile(self, user_visited_listings):
        """사용자 프로파일 생성 (사용자의 방문한 숙소를 기반으로)"""
        # listing_id가 train 데이터에 있는 경우만 필터링
        valid_listings = [lid for lid in user_visited_listings if lid in self.listing['listing_id'].values]
        if not valid_listings:
            raise ValueError("No valid listings found in training data")
            
        visited_features = self.features[self.listing['listing_id'].isin(valid_listings)]
        user_profile = visited_features.mean(axis=0)
        return user_profile

    def get_recommendations_with_user_preference(self, user_visited_listings, topn=10):
        """사용자 프로파일 기반 추천"""
        try:
            user_profile = self.generate_user_profile(user_visited_listings)
            
            # Convert user_profile to a numpy array and reshape it
            user_profile_array = user_profile.to_numpy().reshape(1, -1)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(user_profile_array, self.feature_matrix)
            
            # Get the top indices excluding already visited listings
            all_indices = cosine_sim[0].argsort()[::-1]
            visited_ids = set(user_visited_listings)
            recommended_indices = []
            
            # Filter out already visited listings
            for idx in all_indices:
                listing_id = self.listing.iloc[idx]['listing_id']
                if listing_id not in visited_ids:
                    recommended_indices.append(idx)
                if len(recommended_indices) >= topn:
                    break
            
            # Retrieve recommended listings
            recommended_listings = self.listing.iloc[recommended_indices]
            recommended_ids = recommended_listings['listing_id'].tolist()
            
            return recommended_listings, recommended_ids
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame(), []