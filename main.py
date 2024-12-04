from matplotlib import pyplot as plt
from Processing import DataAdd
from Collaborative_Modeling import ItemBasedModeling
from Collaborative_evaluate import ItemBasedEvaluator
from ContentBased_Modeling import ContentBasedModeling
from ContentBased_evaluate import ContentBasedEvaluator

def main():
    train_dataset = 'data/train_listing_with_visitors.csv'
    test_dataset = 'data/test_listing_with_visitors.csv'

    print("1. Collaborative Filtering")    

    print("evaluation of train data...")
    item_modeling = ItemBasedModeling(train_dataset)
    item_evaluator = ItemBasedEvaluator(item_modeling, k=10)
    mean_precision_score, mae, rmse = item_evaluator.evaluate()
    item_evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("evaluation on test data...")
    item_modeling = ItemBasedModeling(test_dataset)
    item_evaluator = ItemBasedEvaluator(item_modeling, k=10)
    mean_precision_score, mae, rmse = item_evaluator.evaluate()
    # 성능 분포 확인
    print("\nPerformance Distribution:")
    print(stats_df[['precision', 'recall']].describe())

    print(f"Evaluation Results on Random Sample:")
    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}") 

    print("evaluation completed.")
    print("=======================")

    print("\n2. Content-Based Filtering")

    print("evaluation of train data...")
    content_modeling = ContentBasedModeling(train_dataset)
    content_evaluator = ContentBasedEvaluator(content_modeling, k=10)
    avg_precision, avg_recall, stats_df = content_evaluator.evaluate_on_random_sample(sample_size=10, k=10)
    content_evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")

    print("evaluation on test data...")
    content_modeling = ContentBasedModeling(test_dataset)
    content_evaluator = ContentBasedEvaluator(content_modeling, k=10)
    avg_precision, avg_recall, stats_df = content_evaluator.evaluate_on_random_sample(sample_size=10, k=10)
    content_evaluator.print_results(mean_precision_score, mae, rmse)
    print("evaluation completed.")
    print("=======================")


if __name__ == "__main__":
    main()