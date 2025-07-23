import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import zipfile
import requests
from io import BytesIO

# --- Phase 1 & 2: Data Acquisition, Preprocessing & Feature Engineering ---

# For demonstration, we'll download a small MovieLens dataset.
# In a real system, data would be fetched from databases/data lakes.

def download_movielens_100k():
    """Downloads the MovieLens 100k dataset if not already present."""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    data_dir = "./ml-100k"
    if not os.path.exists(data_dir):
        print(f"Downloading MovieLens 100k dataset from {url}...")
        response = requests.get(url)
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(".")
        print("Download complete.")
    else:
        print(f"MovieLens 100k dataset already exists at {data_dir}.")
    return data_dir

data_dir = download_movielens_100k()

# Load data
print("\n--- Loading Data ---")
try:
    # User ratings data
    ratings_df = pd.read_csv(
        os.path.join(data_dir, 'u.data'),
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    print("Ratings Data Sample:")
    print(ratings_df.head())

    # Movie metadata
    # The 'u.item' file is ISO-8859-1 encoded, and columns are pipe-separated
    movies_df = pd.read_csv(
        os.path.join(data_dir, 'u.item'),
        sep='|',
        names=[
            'movie_id', 'movie_title', 'release_date', 'video_release_date',
            'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
            'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ],
        encoding='ISO-8859-1'
    )
    # Drop unnecessary columns for this example
    movies_df = movies_df.drop(columns=['video_release_date', 'IMDb_URL', 'unknown'])
    print("\nMovies Data Sample:")
    print(movies_df.head())

except FileNotFoundError as e:
    print(f"Error loading data: {e}. Please ensure the MovieLens dataset is extracted correctly.")
    exit()

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

# For Content-Based: Create a 'genres' string for each movie
genre_columns = movies_df.columns[5:] # All columns from 'Action' onwards
movies_df['genres_str'] = movies_df[genre_columns].apply(
    lambda row: ' '.join(row.index[row == 1]), axis=1
)
print("\nMovies with combined genres_str:")
print(movies_df[['movie_title', 'genres_str']].head())


# --- Phase 3: Algorithm Selection and Model Development ---

# 1. Popularity-Based Recommendation
print("\n--- Popularity-Based Recommender ---")
# Calculate average rating and count of ratings for each movie
movie_stats = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
movie_stats.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'}, inplace=True)

# Merge with movie titles
popular_movies_df = pd.merge(movie_stats, movies_df[['movie_id', 'movie_title']], on='movie_id')

# Filter for movies with a minimum number of ratings to ensure reliability
min_ratings = 50
qualified_movies = popular_movies_df[popular_movies_df['rating_count'] >= min_ratings]

# Sort by average rating
popular_movies = qualified_movies.sort_values(by='avg_rating', ascending=False)
print(f"\nTop 10 Popular Movies (min {min_ratings} ratings):")
print(popular_movies.head(10))

def get_popular_recommendations(num_recommendations=10):
    """Returns a list of popular movie recommendations."""
    return popular_movies['movie_title'].head(num_recommendations).tolist()

# 2. Content-Based Filtering
print("\n--- Content-Based Recommender ---")
# Create TF-IDF matrix for genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres_str'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping from movie title to index
indices = pd.Series(movies_df.index, index=movies_df['movie_title']).drop_duplicates()

def get_content_based_recommendations(movie_title, cosine_sim_matrix, df, indices, num_recommendations=10):
    """
    Generates content-based recommendations for a given movie title.
    """
    if movie_title not in indices:
        print(f"Movie '{movie_title}' not found in the database.")
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1] # Exclude the movie itself

    movie_indices = [i[0] for i in sim_scores]
    return df['movie_title'].iloc[movie_indices].tolist()

# Example Content-Based Recommendation
example_movie = 'Star Wars (1977)'
print(f"\nContent-based recommendations for '{example_movie}':")
print(get_content_based_recommendations(example_movie, cosine_sim, movies_df, indices))


# 3. Collaborative Filtering (Matrix Factorization - SVD)
print("\n--- Collaborative Filtering (SVD) Recommender ---")

# The Surprise library requires a specific data format
# Define a Reader object with the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the dataset from the pandas DataFrame
# Note: For Surprise, we only need user_id, movie_id, rating columns
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# --- Phase 4: Model Training and Evaluation ---

# Split data into training and test set
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use the SVD algorithm
svd_model = SVD(random_state=42, n_epochs=20, lr_all=0.005, reg_all=0.02) # Example hyperparameters

# Train the model
print("Training SVD model...")
svd_model.fit(trainset)
print("SVD model training complete.")

# Make predictions on the test set
predictions = svd_model.test(testset)

# Evaluate the model
print("\nSVD Model Evaluation:")
accuracy.rmse(predictions, verbose=True)
accuracy.mae(predictions, verbose=True)

# Function to get SVD recommendations for a specific user
def get_svd_recommendations(user_id, movies_df, ratings_df, svd_model, num_recommendations=10):
    """
    Generates SVD-based recommendations for a given user.
    Filters out movies the user has already rated.
    """
    # Get a list of all movie IDs
    all_movie_ids = movies_df['movie_id'].unique()

    # Get movies the user has already rated
    rated_movie_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].unique()

    # Find movies the user hasn't rated
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]

    # Predict ratings for unrated movies
    predicted_ratings = []
    for movie_id in movies_to_predict:
        # svd_model.predict(user_id, movie_id).est gives the estimated rating
        predicted_ratings.append((movie_id, svd_model.predict(user_id, movie_id).est))

    # Sort predictions by estimated rating in descending order
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommendations and merge with movie titles
    top_n_recommendations = predicted_ratings[:num_recommendations]
    recommended_movie_ids = [movie_id for movie_id, _ in top_n_recommendations]

    # Map movie_ids to titles
    recommendations_with_titles = movies_df[movies_df['movie_id'].isin(recommended_movie_ids)]
    # Preserve order of recommendations
    recommendations_with_titles = recommendations_with_titles.set_index('movie_id').loc[recommended_movie_ids].reset_index()

    return recommendations_with_titles[['movie_title', 'movie_id']]

# Example SVD Recommendation for user 1
user_id_example = 1
print(f"\nSVD recommendations for User {user_id_example}:")
svd_recs = get_svd_recommendations(user_id_example, movies_df, ratings_df, svd_model)
print(svd_recs)

# --- Phase 5 & 6: System Architecture, Deployment, Iteration (Conceptual) ---
print("\n--- System Architecture & Deployment (Conceptual) ---")
print("In a real Netflix-like system, the models trained above would be part of a larger architecture:")
print("1.  **API Layer:** A Flask/FastAPI/Django application would expose endpoints for recommendations.")
print("    e.g., `/recommendations?user_id=123`")
print("2.  **Model Serving:** Trained models (like svd_model, or a combination of models) would be loaded and used to generate predictions on demand.")
print("    - Could use Flask/FastAPI, or specialized serving frameworks like TensorFlow Serving, PyTorch Serve.")
print("3.  **Data Pipelines:** Real-time user interactions would feed into a stream processing system (e.g., Kafka).")
print("    - Batch jobs (e.g., Apache Spark, Dask) would periodically retrain models with updated data.")
print("4.  **Database/Data Lake:** User profiles, movie metadata, and interaction history would reside in scalable databases.")
print("5.  **Hybrid Approach Orchestration:** A 'Recommender Orchestrator' would decide which model(s) to use for a given request (e.g., popularity for new users, SVD for active users, content-based for cold-start movies).")
print("6.  **A/B Testing Framework:** To test new recommendation algorithms or changes in real-time.")
print("7.  **Monitoring:** Dashboards to track model performance, latency, and user engagement metrics.")
print("\nThis script demonstrates the core ML logic, but the engineering effort for a production system is substantial.")
print("Consider using cloud services (AWS SageMaker, Google Cloud AI Platform) for easier deployment and scaling.")