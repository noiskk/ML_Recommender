from matplotlib import pyplot as plt
from ContentBased_Modeling import ListingRecommender
from ContentBased_evaluate import RecommenderEvaluator
from ItemBased_Modeling import ItemBasedRecommender
from ItemBased_evaluate import ItemBasedEvaluator
import time

def print_progress(start_time, message):
    """진행 시간 출력 함수"""
    elapsed_time = time.time() - start_time
    print(f"[{elapsed_time:.2f}s] {message}")

def main():
    train_data = 'data/train_listing_with_visitors3.csv'
    test_data = 'data/test_listing_with_visitors.csv'
    total_start_time = time.time()

    print("\n1. Item-Based Filtering")
    print_progress(total_start_time, "Item-Based 모델링 시작...")
    
    # 데이터 로드 및 추천 시스템 초기화
    model_start_time = time.time()
    item_based_recommender = ItemBasedRecommender(test_data)
    print_progress(model_start_time, "Item-Based 모델 초기화 완료")
    
    # 유사도 행렬 계산
    sim_start_time = time.time()
    item_based_recommender.create_similarity_matrix(batch_size=100)
    print_progress(sim_start_time, "유사도 행렬 계산 완료")
    
    # 평가 시작
    eval_start_time = time.time()
    evaluator = ItemBasedEvaluator(item_based_recommender)
    print_progress(eval_start_time, "평가기 초기화 완료")

    # 모델 평가
    print("\n평가 진행 중...")
    precision, recall, detailed_results = evaluator.evaluate_model(
        sample_size=10,
        k=10
    )
    print_progress(eval_start_time, "Item-Based 평가 완료")

    user_id_to_check = 49931038  # 확인하고 싶은 사용자 ID로 변경하세요.
    item_based_recommender.print_recommendations(user_id_to_check, topn=10)

    print("\n2. Content-Based Filtering")
    content_start_time = time.time()
    print_progress(content_start_time, "Content-Based 모델링 시작...")

    # Content-Based 모델 초기화
    recommender_train = ListingRecommender(test_data)
    print_progress(content_start_time, "Content-Based 모델 초기화 완료")

    # 평가기 초기화
    content_eval_time = time.time()
    evaluator = RecommenderEvaluator(recommender_train)
    print_progress(content_eval_time, "Content-Based 평가기 초기화 완료")

    # 평가 진행
    print("\n평가 진행 중...")
    avg_precision, avg_recall, stats_df = evaluator.evaluate_model(
        sample_size=50,
        k=10
    )
    print_progress(content_eval_time, "Content-Based 평가 완료")

    # 전체 실행 시간
    print(f"\n전체 실행 시간: {time.time() - total_start_time:.2f}초")

    # 결과 출력
    print("\n=== 최종 결과 ===")
    print("\nItem-Based Filtering 결과:")
    print(detailed_results[['precision', 'recall']].describe())
    print(f"Precision@10: {precision:.4f}")
    print(f"Recall@10: {recall:.4f}")

    print("\nContent-Based Filtering 결과:")
    print(stats_df[['precision', 'recall']].describe())
    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}")

if __name__ == "__main__":
    main()