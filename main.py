from matplotlib import pyplot as plt
from Processing import DataAdd
from Collaborative_Modeling import Modeling
from Collaborative_evaluate import ModelEvaluator
from ContentBased_Modeling import ListingRecommender
from ContentBased_evaluate import RecommenderEvaluator

def main():
    listing = 'data/train_listing_with_visitors.csv'

    input_data_path = 'data/train_data(v0.1).csv'
    updated_data_path = 'data/train_updated_data_visitors_and_ratings.csv'
    test_data_path = 'data/test_data(v0.1).csv'
    updated_test_data_path = 'data/test_updated_data_visitors_and_ratings.csv'
    listing = 'data/listing_with_visitors.csv'

    print("Train data processing...")
    data_add_train = DataAdd(input_data_path, updated_data_path)
    data_add_train.process_data()
    print("Train data processing completed.")
    print("=======================")

    print("Test data processing...")
    data_add_test = DataAdd(test_data_path, updated_test_data_path)
    data_add_test.process_data()
    print("Test data processing completed.")
    print("=======================")

    print("1. Collaborative Filtering")    

    print("evaluation of train data...")
    modeling = Modeling(updated_data_path)
    evaluator = ModelEvaluator(modeling, k=5)
    mean_precision_score, mae, rmse = evaluator.evaluate()
    evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("evaluation on test data...")
    modeling = Modeling(updated_test_data_path)
    evaluator = ModelEvaluator(modeling, k=5)
    mean_precision_score, mae, rmse = evaluator.evaluate()
    evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("\n2. Content-Based Filtering")

    # ListingRecommender 객체를 생성 (파일 경로를 실제로 지정)
    recommender_train = ListingRecommender(listing)

    # RecommenderEvaluator 객체를 생성
    evaluator = RecommenderEvaluator(recommender_train)

    # 무작위 샘플에 대한 평가
    avg_precision, avg_recall, stats_df = evaluator.evaluate_on_random_sample(sample_size=10, k=10)

    # 성능 분포 확인
    print("\nPerformance Distribution:")
    print(stats_df[['precision', 'recall']].describe())

    print(f"Evaluation Results on Random Sample:")
    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}") 

if __name__ == "__main__":
    main()