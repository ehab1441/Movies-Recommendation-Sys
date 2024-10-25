import streamlit as st

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
    recommendations = recommend_unique_movies(genre, popularity, language, release_era)
    st.write("Recommended Movies:", recommendations)