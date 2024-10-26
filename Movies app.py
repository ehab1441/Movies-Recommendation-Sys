import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('https://github.com/ehab1441/Movies-Recommendation-Sys/blob/main/final_movies_data.csv')

def extract_years(df, col):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df['Release_Year'] = df[col].dt.year
    df.drop(columns=['Release_Date'], inplace=True)
    return df
movies = extract_years(movies, 'Release_Date')

def categorize_popularity(df):
    bins = [0, 200, 500, 700, df['Popularity'].max()]
    labels = ['Low', 'Medium', 'High', 'Very High']
    df['Popularity_'] = pd.cut(df['Popularity'], bins = bins, labels = labels, include_lowest=True)
    df.drop(columns=['Popularity'], inplace=True)
    return df
movies = categorize_popularity(movies)

movies.drop(columns=['Vote_Count'], inplace=True)

def map_language_codes_to_full_names(df):
    language_map = {
        'en': 'English', 'ja': 'Japanese', 'fr': 'French', 'hi': 'Hindi', 'es': 'Spanish',
        'ru': 'Russian', 'de': 'German', 'th': 'Thai', 'ko': 'Korean', 'tr': 'Turkish',
        'cn': 'Chinese', 'zh': 'Chinese', 'it': 'Italian', 'pt': 'Portuguese', 'ml': 'Malayalam',
        'pl': 'Polish', 'fi': 'Finnish', 'no': 'Norwegian', 'da': 'Danish', 'id': 'Indonesian',
        'sv': 'Swedish', 'nl': 'Dutch', 'te': 'Telugu', 'sr': 'Serbian', 'is': 'Icelandic',
        'ro': 'Romanian', 'tl': 'Tagalog', 'fa': 'Persian', 'uk': 'Ukrainian', 'nb': 'Norwegian Bokmål',
        'eu': 'Basque', 'lv': 'Latvian', 'ar': 'Arabic', 'el': 'Greek', 'cs': 'Czech', 'ms': 'Malay',
        'bn': 'Bengali', 'ca': 'Catalan', 'la': 'Latin', 'ta': 'Tamil', 'hu': 'Hungarian', 
        'he': 'Hebrew', 'et': 'Estonian'
    }
    
    df['Original_Language_Full'] = df['Original_Language'].map(language_map)
    df.drop(columns=['Original_Language'], inplace = True)
    
    return df
movies = map_language_codes_to_full_names(movies)

def categorize_year(df):
    bins = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, df['Release_Year'].max()]
    labels = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
    df['Release_Era'] = pd.cut(df['Release_Year'], bins = bins, labels = labels, include_lowest=True)
    return df
movies = categorize_year(movies)

movies = movies.drop_duplicates(subset='Title', keep='first')

label_encoder = LabelEncoder()
movies['Genre_Encoded'] = label_encoder.fit_transform(movies['Genre'])
movies['Title_Encoded'] = label_encoder.fit_transform(movies['Title'])
movies['Popularity_Encoded'] = label_encoder.fit_transform(movies['Popularity_'])
movies['Language_Encoded'] = label_encoder.fit_transform(movies['Original_Language_Full'])
movies['Genre_Encoded'] = label_encoder.fit_transform(movies['Genre_First_Word'])

movies['Release_Era'] = movies['Release_Era'].astype(int)
movies['Genre_First_Word'] = movies['Genre'].str.split().str[0].str.replace(r'[^\w\s]', '', regex=True)

X = movies[['Genre_Encoded', 'Popularity_Encoded', 'Language_Encoded', 'Release_Era']]
​
similarity_matrix = cosine_similarity(X)
​
def recommend_movies(movie_title_encoded, similarity_matrix, movie_titles, top_n=5):
    # Find the index of the encoded movie title
    movie_idx = np.where(movie_titles == movie_title_encoded)[0][0]
    
    # Get the similarity scores for the specified movie
    similar_movies = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort the movies based on similarity score, descending
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies
    recommendations = similar_movies[0:top_n]
    
    # Return the titles of recommended movies
    return [movie_titles[i[0]] for i in recommendations]
​
movie_titles = movies['Title_Encoded'].values

def get_encoded_title_by_features(Genre, Popularity, Language, Release_Era, movies_df):
    # Filter the DataFrame based on provided feature values
    filtered_movies = movies_df[
        (movies_df['Genre_First_Word'] == Genre) &
        (movies_df['Popularity_'] == Popularity) &
        (movies_df['Original_Language_Full'] == Language) &
        (movies_df['Release_Era'] == Release_Era)
    ]
    
    
    if not filtered_movies.empty:
        return filtered_movies['Title_Encoded'].values
    else:
        return "No movie found with the specified feature values."

encoded_titles = get_encoded_title_by_features(
    genre,
    popularity,
    language,
    release_era,
    movies
)

some_movie_encoded = encoded_titles[0]
recommended_movies = recommend_movies(some_movie_encoded, similarity_matrix, movie_titles, top_n=5)

def get_specific_movie_details_by_encoded_title(title_encoded, movies_df, columns):
    # Filter the DataFrame to find the row with the specified Title_Encoded
    movie_details = movies_df[movies_df['Title_Encoded'] == title_encoded]
    
    # Check if any movie details are found
    if not movie_details.empty:
        # Return the specified columns
        return movie_details[columns].iloc[0]  # Return the first matching row as a Series
    else:
        return None  # Return None if no movie is found

# Example usage
# Replace these with actual encoded titles from your dataset
encoded_titles_input = recommended_movies  # Example values for Title_Encoded

# Specify the columns you want to retrieve
columns_to_retrieve = ['Title', 'Overview','Genre', 'Popularity_', 'Vote_Average', 'Poster_Url']

# Loop through each encoded title and get the movie details
for title_encoded_input in encoded_titles_input:
    movie_info = get_specific_movie_details_by_encoded_title(title_encoded_input, movies, columns_to_retrieve)
    
    # Print the specific movie details
    if movie_info is not None:
        for col in columns_to_retrieve:
            print(f"{col}: {movie_info[col]}")
    else:
        print(f"No movie found with Title Encoded {title_encoded_input}.")

genre = st.selectbox("Genre", options=['Action', 'Crime', 'Thriller', 'Animation', 'Horror', 'Science',
       'Fantasy', 'Romance', 'Drama', 'Western', 'Family', 'Comedy',
       'Adventure', 'Mystery', 'TV', 'Documentary', 'War', 'Music',
       'History'])  # add your options here
popularity = st.selectbox("Popularity", options=['Very High', 'High', 'Medium', 'Low'])
language = st.selectbox("Language", options=['English', 'Japanese', 'French', 'Hindi', 'Spanish', 'Russian',
       'German', 'Thai', 'Korean', 'Turkish', 'Chinese', 'Italian',
       'Portuguese', 'Malayalam', 'Polish', 'Finnish', 'Norwegian',
       'Danish', 'Indonesian', 'Swedish', 'Dutch', 'Telugu', 'Serbian',
       'Icelandic', 'Romanian', 'Tagalog', 'Persian', 'Ukrainian',
       'Norwegian Bokmål', 'Basque', 'Latvian', 'Arabic', 'Greek',
       'Czech', 'Malay', 'Bengali', 'Catalan', 'Latin', 'Tamil',
       'Hungarian', 'Hebrew', 'Estonian'])
release_era = st.number_input("Release Era", min_value=1900, max_value=2023)

if st.button("Get Recommendations"):
    recommended_movies = recommend_movies(some_movie_encoded, similarity_matrix, movie_titles, top_n=5)
    st.write("Recommended Movies:", recommended_movies)