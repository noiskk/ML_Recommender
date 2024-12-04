from matplotlib import pyplot as plt
from Collaborative_Modeling import ItemBasedModeling
from Collaborative_evaluate import ItemBasedEvaluator
from ContentBased_Modeling import ContentBasedModeling
from ContentBased_evaluate import ContentBasedEvaluator

def main():
    train_dataset = 'data/train_listing_with_visitors.csv'
    test_dataset = 'data/test_listing_with_visitors.csv'

    print("1. Collaborative Filtering")    
    # train 데이터로 모델 학습
    item_modeling = ItemBasedModeling(train_dataset)
    item_modeling.create_similarity_matrix()  # 유사도 행렬 생성
    
    # test 데이터로 평가
    print("Evaluating on test data...")
    test_evaluator = ItemBasedEvaluator(item_modeling, k=10, test_file=test_dataset)
    results_df = test_evaluator.evaluate_model(sample_size=100)
    
    print("\nPerformance Distribution:")
    print(results_df[['precision', 'recall']].describe())
    print(f"\nEvaluation Results:")
    print(f"Average Precision@{10}: {results_df['precision'].mean():.4f}")
    print(f"Average Recall@{10}: {results_df['recall'].mean():.4f}")
    print("evaluation completed.")
    print("=======================")

    print("\n2. Content-Based Filtering")
    # train 데이터로 모델 학습
    content_modeling = ContentBasedModeling(train_dataset)
    
    # test 데이터로 평가
    print("Evaluating on test data...")
    test_evaluator = ContentBasedEvaluator(content_modeling, k=10, test_file=test_dataset)
    results_df = test_evaluator.evaluate_model(sample_size=100)
    
    print("\nPerformance Distribution:")
    print(results_df[['precision', 'recall']].describe())
    print(f"\nEvaluation Results:")
    print(f"Average Precision@{10}: {results_df['precision'].mean():.4f}")
    print(f"Average Recall@{10}: {results_df['recall'].mean():.4f}")
    print("evaluation completed.")
    print("=======================")

if __name__ == "__main__":
    main()