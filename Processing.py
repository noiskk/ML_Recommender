import pandas as pd
import numpy as np

class DataAdd:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.visitor_pool = [f"visitor_{i}" for i in range(1, 101)]
        self.data = pd.read_csv(self.input_path)

    def assign_sparse_visitors(self, num_visitors=4):
        self.data['visitors'] = self.data.apply(
            lambda row: np.random.choice(self.visitor_pool, num_visitors, replace=False).tolist(), axis=1
        )

    def assign_varied_user_ratings(self, variation_range=(-5, 5)):
        self.data['user_ratings'] = self.data.apply(
            lambda row: [
                max(0, min(100, int(row['review_scores_rating'] + np.random.randint(*variation_range))))
                for _ in range(len(row['visitors']))
            ], axis=1
        )

    def save_updated_data(self):
        self.data.to_csv(self.output_path, index=False)

    def process_data(self):
        self.assign_sparse_visitors()
        self.assign_varied_user_ratings()
        self.save_updated_data()
        print(f"Updated data saved to {self.output_path}")

# main function for DataAdd
def main_data_add():
    input_path = 'data/train_data(v0.1).csv'
    output_path = 'data/updated_data_visitors_and_ratings.csv'
    data_add = DataAdd(input_path, output_path)
    data_add.process_data()

if __name__ == "__main__":
    main_data_add()