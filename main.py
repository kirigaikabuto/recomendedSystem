import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("data.csv")

test_data = data["название"].iloc[22]

tfidf = TfidfVectorizer()
data['Описание'] = data['Описание'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['название'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['название']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    topMovies = data['название'].iloc[movie_indices]
    movie_ratings = [i[1] for i in sim_scores]
    topMovies['Score'] = movie_ratings
    return topMovies


result = get_recommendations(test_data)
print("результаты")
print(result)
