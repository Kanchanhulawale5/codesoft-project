import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction data (replace this with your actual dataset)
data = {
    'user_id': [1, 1, 2, 2, 3],
    'item_id': ['A', 'B', 'A', 'C', 'B'],
    'rating': [5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

# Create user-item matrix (pivot table)
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

# Calculate user-user similarity (cosine similarity)
user_similarity = cosine_similarity(user_item_matrix)

# Function to recommend items for a given user ID
def recommend_items_for_user(user_id):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found in the dataset.")
        return

    target_user_ratings = user_item_matrix.loc[user_id]

    # Find similar users
    similar_users = user_similarity[user_id - 1]  # user_id is 1-indexed

    # Identify items not interacted by the target user
    items_not_interacted = target_user_ratings[target_user_ratings == 0].index

    # Recommend top items based on similar users
    recommendations = []
    for user_id, similarity in enumerate(similar_users):
        if user_id + 1 != user_id:  # Exclude the target user itself
            similar_user_ratings = user_item_matrix.loc[user_id + 1]
            recommended_items = similar_user_ratings[items_not_interacted]
            recommendations.extend(recommended_items[recommended_items > 0].index)

    # Display recommended items
    recommended_items = list(set(recommendations))[:5]
    print(f"Top recommended items for user {user_id}: {recommended_items}")

# Get user ID from input
try:
    user_id = int(input("Enter user ID: "))
    recommend_items_for_user(user_id)
except ValueError:
    print("Please enter a valid integer for user ID.")