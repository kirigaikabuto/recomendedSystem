import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import pickle

data = pd.read_csv("data.csv")
test_data = data["название"].iloc[23]
tfidf_matrix = pickle.load(open('tfidf_matrix.pickle', 'rb'))


def get_recommendations(title, tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['название']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    topMovies = data['название'].iloc[movie_indices]
    movie_ratings = [i[1] for i in sim_scores]
    result = []
    for i in range(len(topMovies)):
        anime = {}
        anime["name"] = topMovies.iloc[i]
        anime["score"] = movie_ratings[i]
        result.append(anime)
    return result

result = get_recommendations(test_data, tfidf_matrix)
print(f"Результаты по рекомендации на аниме {test_data}")
print(result)
