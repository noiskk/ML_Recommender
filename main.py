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

def evaluate_item_based(dataset):
    print("\n1. Item-Based Filtering")
    print_progress(time.time(), "Item-Based 모델링 시작...")
    
    # 데이터 로드 및 추천 시스템 초기화
    model_start_time = time.time()
    item_based_recommender = ItemBasedRecommender(dataset)
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
        sample_size=20,
        k=30
    )
    print_progress(eval_start_time, "Item-Based 평가 완료")

    return precision, recall, detailed_results

def evaluate_content_based(dataset):
    print("\n2. Content-Based Filtering")
    content_start_time = time.time()
    print_progress(content_start_time, "Content-Based 모델링 시작...")

    # Content-Based 모델 초기화
    recommender_train = ListingRecommender(dataset)
    print_progress(content_start_time, "Content-Based 모델 초기화 완료")

    # 평가기 초기화
    content_eval_time = time.time()
    evaluator = RecommenderEvaluator(recommender_train)
    print_progress(content_eval_time, "Content-Based 평가기 초기화 완료")

    # 평가 진행
    print("\n평가 진행 중...")
    avg_precision, avg_recall, stats_df = evaluator.evaluate_model(
        sample_size=20,
        k=30
    )
    print_progress(content_eval_time, "Content-Based 평가 완료")

    return avg_precision, avg_recall, stats_df

def main():
    dataset = 'data/listing_with_visitors.csv'

    # Item-Based 평가
    precision, recall, detailed_results = evaluate_item_based(dataset)

    # Content-Based 평가
    avg_precision, avg_recall, stats_df = evaluate_content_based(dataset)

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