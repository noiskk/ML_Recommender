import pandas as pd

class Preprocessing:
  def __init__(self, listing_path, review_path):
      self.listing_path = listing_path
      self.review_path = review_path

  def add_visitors_to_listing(listing_path, review_path):
      # 데이터 파일 읽기
      listing_df = pd.read_csv(listing_path)
      review_df = pd.read_csv(review_path)

      # listing_id 별로 reviewer_id를 그룹화하여 리스트로 변환
      visitors_by_listing = review_df.groupby('listing_id')['reviewer_id'].agg(lambda x: x.tolist()).reset_index()
      visitors_by_listing = visitors_by_listing.rename(columns={'reviewer_id': 'visitors'})

      # listing 데이터프레임에 visitors 컬럼 추가
      listing_df = listing_df.merge(visitors_by_listing, on='listing_id', how='left')

      # visitors 컬럼이 NaN인 경우 빈 리스트로 채우기
      listing_df['visitors'] = listing_df['visitors'].fillna('').apply(lambda x: [] if x == '' else x)

      return listing_df