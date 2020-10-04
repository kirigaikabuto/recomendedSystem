import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
data = pd.read_csv("data.csv")

test_data = data["название"].iloc[22]

tfidf = TfidfVectorizer()
data['Описание'] = data['Описание'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['название'])
pickle.dump(tfidf_matrix, open('../RecommendationApi/tfidf_matrix.pickle', 'wb'))

