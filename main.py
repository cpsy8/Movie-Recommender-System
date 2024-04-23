import pandas as pd
import numpy as np
import ast
import nltk
import requests

url = "https://drive.google.com/uc?id=1Mv4yi1ycNxVTlNuX1XAoAqXzDS-psQYw&export=download"
response = requests.get(url)
with open('movies.csv', 'wb') as file:
    file.write(response.content)

url = "https://drive.google.com/uc?id=1cfLmqsXLHVBZRBIDpebZp44H0ORqDAhB&export=download"
response = requests.get(url)
with open('credits.csv', 'wb') as file:
    file.write(response.content)

movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')


movies = movies.merge(credits, on='title') #merging movies and credits file using the title label
movies = movies[['movie_id','title','overview','genres', 'keywords', 'cast', 'crew']] #movies has now only these columns
movies.isnull().sum() #check if there is any missing value in any column
movies.dropna(inplace=True) #remove those rows having msssing values
movies.duplicated().sum() # check if there is any duplicate value in the data

"""The idea is to create a paragraph for every movie and then check the matching words for silmilar movies, its kind of documents similarity checking, 
so now will merge all the column together to convert them into a single paragraph"""

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

#we will use a helper function to convert the string to a dictionary or the original data inside that string
def convert(obj):
  L = []
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

def convert3(obj):
  L = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter != 3 :
      L.append(i['name'])
      counter += 1
    else:
      break
  return L

def fetch_dir(obj):
  l = []
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      l.append(i['name'])
      break
  return l

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return ' '.join(y)

movies['keywords'] = movies['keywords'].apply(convert)
movies['genres'] = movies['genres'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_dir)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df['tags'] = new_df['tags'].apply(stem)

vectors = cv.fit_transform(new_df['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  if movie_index == 0:
    print('Invalid request')
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

  for i in movies_list:
    return new_df.iloc[i[0]].title

import pickle
pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

