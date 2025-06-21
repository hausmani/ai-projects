import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample movie dataset
data = {
    'title': [
        'The Avengers',
        'Avengers: Age of Ultron',
        'Iron Man',
        'The Dark Knight',
        'Batman Begins',
        'Man of Steel',
        'Superman Returns'
    ],
    'description': [
        'superhero team saves world',
        'heroes unite against robot',
        'billionaire builds suit',
        'vigilante fights crime',
        'origin of batman',
        'alien hero protects earth',
        'superman comes back'
    ]
}

df = pd.DataFrame(data)

# CountVectorizer()	Turns movie descriptions into word frequency vectors.
# cosine_similarity()	Measures how similar two vectors (movies) are based on word overlap.

# Why?
# This is not supervised learning
# We're not predicting a label or number — we're comparing vectors to find similar items
# Cosine similarity is perfect for this

# Convert text to vectors
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(df['description'])

# Compute similarity
similarity = cosine_similarity(vectors)


# Recommend function
def recommend(title):
    if title not in df['title'].values:
        return "Movie not found."
    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]  # top 3 excluding itself
    recommended_titles = [df.iloc[i[0]]['title'] for i in scores]
    return recommended_titles


# Test
print("Recommended for 'Iron Man':", recommend('Iron Man'))
