import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata = pd.read_csv('recomend.csv', low_memory=False)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['rating'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(metadata.index, index=metadata['name'])


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:5]

    movie_indices = [i[0] for i in sim_scores]

    return metadata['name'].iloc[movie_indices], metadata['year'].iloc[movie_indices]


names, practice = get_recommendations('john')

a = dict(zip(names, practice))

sorted_dict = {}
sorted_keys = sorted(a, key=a.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = a[w]

print(sorted_dict)