import streamlit as st

genre = st.selectbox("Genre", options=['Action', 'Crime', 'Thriller', 'Animation', 'Horror', 'Science',
       'Fantasy', 'Romance', 'Drama', 'Western', 'Family', 'Comedy',
       'Adventure', 'Mystery', 'TV', 'Documentary', 'War', 'Music',
       'History'])  # add your options here
popularity = st.selectbox("Popularity",  options=['Very High', 'High', 'Medium', 'Low'])
language = st.selectbox("Language", options=['English', 'Japanese', 'French', 'Hindi', 'Spanish', 'Russian',
       'German', 'Thai', 'Korean', 'Turkish', 'Chinese', 'Italian',
       'Portuguese', 'Malayalam', 'Polish', 'Finnish', 'Norwegian',
       'Danish', 'Indonesian', 'Swedish', 'Dutch', 'Telugu', 'Serbian',
       'Icelandic', 'Romanian', 'Tagalog', 'Persian', 'Ukrainian',
       'Norwegian Bokmål', 'Basque', 'Latvian', 'Arabic', 'Greek',
       'Czech', 'Malay', 'Bengali', 'Catalan', 'Latin', 'Tamil',
       'Hungarian', 'Hebrew', 'Estonian'])
st.selectbox("Release Era", options=[1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

if st.button("Get Recommendations"):
    recommendations = recommend_movies(genre, popularity, language, release_era)
    st.write("Recommended Movies:", recommendations)
