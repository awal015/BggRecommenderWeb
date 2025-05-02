import os
import pickle
import pandas as pd
from google.cloud import storage
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template, jsonify
import time
from scipy.sparse import csr_matrix

app = Flask(__name__)
BUCKET_NAME = "bggrecommenderv2"
STORAGE_CLIENT = storage.Client()
BUCKET = STORAGE_CLIENT.bucket(BUCKET_NAME)


def load_pickle_from_gcs(filename):
    """Loads a pickle file from Google Cloud Storage.

    Args:
        filename (str): The name of the pickle file.

    Returns:
        object: The unpickled object, or None on error.
    """
    start_time = time.time()
    try:
        blob = BUCKET.blob(filename)
        data = blob.download_as_bytes()
        obj = pickle.loads(data)
        print(f"Loaded {filename} from GCS in {time.time() - start_time:.2f} seconds")
        return obj
    except Exception as e:
        print(f"Error loading {filename} from GCS: {e}")
        return None


def load_data():
    global tfidf_matrix, user_item_matrix, games, games_cat, tfidf_matrix_cat

    start_time = time.time()

    tfidf_matrix = load_pickle_from_gcs("tfidf_matrix.pkl")  # Load from Pickle
    user_item_matrix = load_pickle_from_gcs("user_item_matrix.pkl")  # Load from Pickle
    games = load_pickle_from_gcs("games.pkl")  # Load from Pickle
    games_cat = load_pickle_from_gcs("games_cat.pkl")
    tfidf_matrix_cat = load_pickle_from_gcs("tfidf_matrix_cat.pkl")

    end_time = time.time()

    if not all([tfidf_matrix is not None, user_item_matrix is not None, games is not None, games_cat is not None,
                tfidf_matrix_cat is not None]):
        print(f"Failed to load data from GCS in {end_time - start_time:.2f} seconds")
        return False

    print(f"Loaded all data from GCS in {end_time - start_time:.2f} seconds")
    return True


# Load data at startup
if not load_data():
    print("Failed to load data, exiting")
    exit()


# ------------------------------
# 1. Recommendation Functions
# ------------------------------
def recommend_games(user_item_matrix, active_user_id, num_recommendations=10):
    if active_user_id not in user_item_matrix.index:
        print(f"User {active_user_id} not found.")
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
    if active_user_id not in user_item_matrix.index:
        print(f"User {active_user_id} not found.")
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
        for game_id, rating in unrated_games.sort_values(ascending=False).items():
            if game_id not in recommended_games:
                recommended_games.append(game_id)
            if len(recommended_games) >= num_recommendations:
                break
        if len(recommended_games) >= num_recommendations:
            break
    return recommended_games[:num_recommendations]


def content_based_recommendations(game_title, tfidf_matrix, games, top_n=10):
    idx = games[games['Name'] == game_title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n - 1:-1]
    related_docs_indices = related_docs_indices[related_docs_indices != idx]
    return games['Name'].iloc[related_docs_indices].tolist()


def content_based_recommendations_cat(game_title, tfidf_matrix, games_cat, top_n=10):
    idx = games_cat[games_cat['Name'] == game_title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix.loc[[idx]], tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n - 1:-1]
    related_docs_indices = related_docs_indices[related_docs_indices != idx]
    return games_cat['Name'].iloc[related_docs_indices].tolist()


def calc_similarity(similar_user_ratings, active_user_ratings):
    common_ratings = similar_user_ratings.merge(
        active_user_ratings, how='inner', on='BGGId',
        suffixes=('_similar', '_active'))
    if not common_ratings.empty:
        cos_distance = distance.cosine(common_ratings['Rating_similar'],
                                       common_ratings['Rating_active'])
        return 1 - cos_distance
    else:
        return 0


def hybrid_recommendations(user_id, game_title, user_item_matrix, tfidf_matrix,
                           tfidf_matrix_cat, games, top_n=25):
    collab_recommendations = recommend_games_knn(user_item_matrix, user_id)
    content_recommendations = content_based_recommendations(
        game_title, tfidf_matrix, games, top_n)
    content_cat_recommendations = content_based_recommendations_cat(
        game_title, tfidf_matrix, games_cat, top_n)
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


# Load data at startup
if not load_data():
    print("Failed to load data, exiting")
    exit()

# ------------------------------
# 2. Flask Routes
# ------------------------------
active_user_id = ''
user_title = ''


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input_type = request.form['input_type']
        user_name = request.form.get('user_name')
        game_title = request.form.get('game_title')
        if user_input_type == 'user':
            recommendations = recommend_games_knn(user_item_matrix, user_name)
            if not recommendations:
                return render_template('index.html',
                                       message=f"No recommendations for {user_name}")
            game_names = [
                games.loc[games['BGGId'] == bgg_id, 'Name'].values[0]
                for bgg_id in recommendations
            ]
            return render_template('index.html',
                                   recommendations=game_names,
                                   user_name=user_name,
                                   source_type='user')
        elif user_input_type == 'game':
            recommendations = hybrid_recommendations(
                active_user_id, game_title, user_item_matrix, tfidf_matrix,
                tfidf_matrix_cat, games)
            if not recommendations:
                return render_template('index.html',
                                       message=f"No recommendations for {game_title}")
            return render_template('index.html',
                                   recommendations=recommendations,
                                   game_title=game_title,
                                   source_type='game')
        elif user_input_type == 'both':
            recommendations = hybrid_recommendations(
                user_name, game_title, user_item_matrix, tfidf_matrix,
                tfidf_matrix_cat, games)
            if not recommendations:
                return render_template(
                    'index.html',
                    message=
                    f"No recommendations found for user: {user_name} and game title:{game_title}")
            return render_template('index.html',
                                   recommendations=recommendations,
                                   user_name=user_name,
                                   game_title=game_title,
                                   source_type='hybrid')
        else:
            return render_template('index.html',
                                   message="Invalid input. Please enter 'user' or 'game'.")
    return render_template('index.html')


@app.route('/recommendations_json', methods=['POST'])
def get_recommendations_json():
    user_input_type = request.form['input_type']
    user_name = request.form.get('user_name')
    game_title = request.form.get('game_title')
    if user_input_type == 'user':
        recommendations = recommend_games_knn(user_item_matrix, user_name)
        if not recommendations:
            return jsonify({
                'error': f"No recommendations found for user: {user_name}"
            })
        game_names = [
            games.loc[games['BGGId'] == bgg_id, 'Name'].values[0]
            for bgg_id in recommendations
        ]
        return jsonify({'user': user_name,
                        'recommendations': game_names})
    elif user_input_type == 'game':
        recommendations = hybrid_recommendations(
            active_user_id, game_title, user_item_matrix, tfidf_matrix,
            tfidf_matrix_cat, games)
        if not recommendations:
            return jsonify({
                'error': f"No recommendations found for game: {game_title}"
            })
        return jsonify({'game': game_title,
                        'recommendations': recommendations})
    elif user_input_type == 'both':
        recommendations = hybrid_recommendations(
            user_name, game_title, user_item_matrix, tfidf_matrix,
            tfidf_matrix_cat, games)
        if not recommendations:
            return jsonify({
                'error':
                    f"No recommendations found for user: {user_name} and game: {game_title}"
            })
        return jsonify(
            {'user': user_name,
             'game': game_title,
             'recommendations': recommendations})
    else:
        return jsonify({'error': "Invalid input. Please enter 'user' or 'game'."})


if __name__ == "__main__":
    # Use the port that Cloud Run provides
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)

