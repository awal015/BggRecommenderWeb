import pickle
import pandas as pd
from google.cloud import storage
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template, jsonify
app = Flask(__name__)

# Cloud Storage configuration
BUCKET_NAME = "bggrecommenderv2"
STORAGE_CLIENT = storage.Client()  # Initialize the client
BUCKET = STORAGE_CLIENT.bucket(BUCKET_NAME)  # Get the bucket object

# Function to load a pickle file from Cloud Storage
def load_pickle_from_gcs(filename):
    """
    Loads a pickle file from Google Cloud Storage.

    Args:
        filename (str): The name of the pickle file in the bucket.

    Returns:
        object: The unpickled object, or None on error.
    """
    try:
        blob = BUCKET.blob(filename)  # Get the Blob object for the file
        data = blob.download_as_bytes()  # Download the file as bytes
        return pickle.loads(data)  # Unpickle and return the data
    except Exception as e:
        print(f"Error loading {filename} from GCS: {e}")
        return None

# Load pickled files from Cloud Storage
tfidf = load_pickle_from_gcs("tfidf.pkl")
tfidf_matrix = load_pickle_from_gcs("tfidf_matrix.pkl")
tfidf_cat = load_pickle_from_gcs("tfidf_cat.pkl")
tfidf_matrix_cat = load_pickle_from_gcs("tfidf_matrix_cat.pkl")
user_item_matrix = load_pickle_from_gcs("user_item_matrix.pkl")
games = load_pickle_from_gcs("games.pkl")
games_cat = load_pickle_from_gcs("games_cat.pkl")

# Check if any of the loads failed
if not all([tfidf, tfidf_matrix, tfidf_cat, tfidf_matrix_cat, user_item_matrix, games, games_cat]):
    print("Failed to load one or more pickle files from GCS. Exiting.")
    exit()  # Exit the application if any file fails to load.  Important for a web app.

# ------------------------------
# 1. Recommendation Functions (same as in notebook)
# ------------------------------

def recommend_games(user_item_matrix, active_user_id, num_recommendations=10):
    """
    Recommends games using user-based collaborative filtering.
    """
    if active_user_id not in user_item_matrix.index:
        print(f"User {active_user_id} not found in the user-item matrix.")
        return []

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = user_similarity_df[active_user_id].sort_values(ascending=False)[1:]

    recommended_games = []
    for similar_user, similarity_score in similar_users.items():
        similar_user_ratings = user_item_matrix.loc[similar_user]
        unrated_games = similar_user_ratings[user_item_matrix.loc[active_user_id] == 0]
        for game_id, rating in unrated_games.sort_values(ascending=False).items():
            if game_id not in recommended_games:
                recommended_games.append(game_id)
            if len(recommended_games) >= num_recommendations:
                break
        if len(recommended_games) >= num_recommendations:
            break

    return recommended_games[:num_recommendations]


def recommend_games_knn(user_item_matrix, active_user_id, num_recommendations=10, k=5):
    """
    Recommends games using KNN-based collaborative filtering.
    """
    if active_user_id not in user_item_matrix.index:
        print(f"User {active_user_id} not found in the user-item matrix.")
        return []

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=-1)
    model_knn.fit(user_item_matrix)

    distances, indices = model_knn.kneighbors(user_item_matrix.loc[[active_user_id]])
    indices = indices.flatten()[1:]
    distances = distances.flatten()[1:]

    recommended_games = []
    for i, similar_user_index in enumerate(indices):
        similar_user_id = user_item_matrix.index[similar_user_index]
        similar_user_ratings = user_item_matrix.loc[similar_user_id]
        unrated_games = similar_user_ratings[user_item_matrix.loc[active_user_id] == 0]
        weighted_ratings = unrated_games * (1 - distances[i])

        for game_id, rating in weighted_ratings.sort_values(ascending=False).items():
            if game_id not in recommended_games:
                recommended_games.append(game_id)
            if len(recommended_games) >= num_recommendations:
                break
        if len(recommended_games) >= num_recommendations:
            break

    return recommended_games[:num_recommendations]


def content_based_recommendations(game_title, tfidf_matrix, games, top_n=10):
    """
    Recommends games based on description similarity.
    """
    idx = games[games['Name'] == game_title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    related_docs_indices = related_docs_indices[related_docs_indices != idx]
    return games['Name'].iloc[related_docs_indices].tolist()


def content_based_recommendations_cat(game_title, tfidf_matrix, games_cat, top_n=10):
    """
    Recommends games based on category similarity.
    """
    idx = games[games_cat['Name'] == game_title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    related_docs_indices = related_docs_indices[related_docs_indices != idx]
    return games_cat['Name'].iloc[related_docs_indices].tolist()

def calc_similarity(similar_user_ratings, active_user_ratings):
    """Calculates the cosine similarity between two users' ratings."""
    common_ratings = similar_user_ratings.merge(
        active_user_ratings,
        how='inner',
        on='BGGId',
        suffixes=('_similar', '_active')
    )
    if not common_ratings.empty:
        cos_distance = distance.cosine(
            common_ratings['Rating_similar'],
            common_ratings['Rating_active']
        )
        return 1 - cos_distance
    else:
        return 0

def hybrid_recommendations(user_id, game_title, user_item_matrix, tfidf_matrix, tfidf_matrix_cat, games, top_n=25):
    """
    Hybrid recommendation function combining collaborative and content-based filtering.
    """
    collab_recommendations = recommend_games_knn(user_item_matrix, user_id)
    content_recommendations = content_based_recommendations(game_title, tfidf_matrix, games, top_n)
    content_cat_recommendations = content_based_recommendations_cat(game_title, tfidf_matrix, games)

    hybrid_recs = []
    seen = set()

    for rec in collab_recommendations:
        game_name = games[games['BGGId'] == rec]['Name'].values
        if game_name.size > 0 and game_name[0] not in seen:
            hybrid_recs.append(game_name[0])
            seen.add(game_name[0])

    for rec in content_cat_recommendations:
        if rec not in seen and len(hybrid_recs) < top_n:
            hybrid_recs.append(rec)
            seen.add(rec)

    return hybrid_recs[:top_n]


# ------------------------------
# 2. Flask Routes
# ------------------------------
active_user_id = '' # Added the active user.
user_title = ''

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page of the web application.  Handles user input and displays recommendations.
    """
    if request.method == 'POST':
        user_input_type = request.form['input_type']
        user_name = request.form.get('user_name')  # Get username
        game_title = request.form.get('game_title')  # Get game title

        if user_input_type == 'user':
            recommendations = recommend_games_knn(user_item_matrix, user_name)
            if not recommendations:
                return render_template('index.html', message=f"No recommendations found for user: {user_name}")
            game_names = [games.loc[games['BGGId'] == bgg_id, 'Name'].values[0] for bgg_id in recommendations]
            return render_template('index.html',  recommendations=game_names, user_name=user_name, source_type='user')

        elif user_input_type == 'game':
            recommendations = hybrid_recommendations(active_user_id, game_title, user_item_matrix, tfidf_matrix, tfidf_matrix_cat, games)
            if not recommendations:
                return render_template('index.html', message=f"No recommendations found for game: {game_title}")
            return render_template('index.html', recommendations=recommendations, game_title=game_title, source_type='game')
        elif user_input_type == 'both':
            recommendations = hybrid_recommendations(user_name, game_title, user_item_matrix, tfidf_matrix, tfidf_matrix_cat, games)
            if not recommendations:
                return render_template('index.html', message=f"No recommendations found for user: {user_name} and game title: {game_title}")
            return render_template('index.html', recommendations=recommendations, user_name=user_name, game_title=game_title, source_type='hybrid')
        else:
            return render_template('index.html', message="Invalid input. Please enter 'user' or 'game'.")

    return render_template('index.html')  # Render the initial form


@app.route('/recommendations_json', methods=['POST'])
def get_recommendations_json():
    """
    Returns recommendations in JSON format.  Useful for AJAX requests or other applications.
    """
    user_input_type = request.form['input_type']
    user_name = request.form.get('user_name')
    game_title = request.form.get('game_title')

    if user_input_type == 'user':
        recommendations = recommend_games_knn(user_item_matrix, user_name)
        if not recommendations:
            return jsonify({'error': f"No recommendations found for user: {user_name}"})
        game_names = [games.loc[games['BGGId'] == bgg_id, 'Name'].values[0] for bgg_id in recommendations]
        return jsonify({'user': user_name, 'recommendations': game_names})
    elif user_input_type == 'game':
        recommendations = hybrid_recommendations(active_user_id, user_title, user_item_matrix, tfidf_matrix, tfidf_matrix_cat, games)
        if not recommendations:
             return jsonify({'error': f"No recommendations found for game: {game_title}"})
        return jsonify({'game': game_title, 'recommendations': recommendations})
    elif user_input_type == 'both':
        recommendations = hybrid_recommendations(user_name, game_title, user_item_matrix, tfidf_matrix, tfidf_matrix_cat, games)
        if not recommendations:
            return jsonify({'error': f"No recommendations found for user: {user_name} and game: {game_title}"})
        return jsonify({'user': user_name, 'game': game_title, 'recommendations': recommendations})
    else:
        return jsonify({'error': "Invalid input. Please enter 'user' or 'game'."})



if __name__ == "__main__":
    app.run(debug=True)