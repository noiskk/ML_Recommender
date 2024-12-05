import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from ast import literal_eval

class ListingRecommender:
    def __init__(self, filepath):
        print("데이터 로딩 중...")
        listing = pd.read_csv(filepath)
        listing['visitors'] = listing['visitors'].apply(literal_eval)
        print(f"데이터 로드 완료: {len(listing)} 개의 리스팅")
        
        print("특성 처리 중...")
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
        
        print("원-핫 인코딩 처리 중...")
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
        print("특성 매트릭스 생성 완료")
        
    def generate_user_profile(self, user_visited_listings):
        """사용자 프로파일 생성 (사용자의 방문한 숙소를 기반으로)"""
        visited_features = self.features[self.listing['listing_id'].isin(user_visited_listings)]
        user_profile = visited_features.mean(axis=0)
        return user_profile

    def get_recommendations_with_user_preference(self, user_visited_listings, topn=10):
        """사용자 프로파일 기반 추천"""
        user_profile = self.generate_user_profile(user_visited_listings)
        
        # Convert user_profile to a numpy array and reshape it
        user_profile_array = user_profile.to_numpy().reshape(1, -1)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(user_profile_array, self.feature_matrix)
        
        # Get the top indices
        top_indices = cosine_sim[0].argsort()[-topn:][::-1]
        
        # Retrieve recommended listings
        recommended_listings = self.listing.iloc[top_indices]
        recommended_ids = recommended_listings['listing_id'].tolist()
        
        return recommended_listings, recommended_ids