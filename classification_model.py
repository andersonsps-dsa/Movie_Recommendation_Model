import pickle
import streamlit as st
import pandas as pd

data_load_state = st.text('Data loading in progress... ⟳')
movie_piv = pickle.load(open('movie.pkl','rb'))
model_ = pickle.load(open('recommend_movie.pkl','rb'))
data_load_state.text('Data uploaded successfully! ✅')

st.title('🎥 Movie Recommendation Model ')
st.write('➤ Please select the film title from the fields below 🎬:')

col1, col2 = st.columns(2)
with col1:
    movie_titles = movie_piv.index.tolist()
    
    def filter_titles(input_text, titles_list):
        filtered_titles = [title for title in titles_list if input_text.lower() in title.lower()]
        return sorted(filtered_titles)
    user_input = st.text_input('Enter a movie title to view suggestions:')
    if user_input:
        suggestions = filter_titles(user_input, movie_titles)
        if suggestions:
            st.write('Suggestions:')
            for suggestion in suggestions:
                st.write(suggestion)
        else:
            st.write('No movies found. Please try another title.')

with col2:
    if user_input and suggestions:
        selected_movie = suggestions[0]
        distances, neighbors = model_.kneighbors(movie_piv.filter(items=[selected_movie], axis=0).values.reshape(1, -1))
        st.write(f'Similar movies to "{selected_movie}":')
        for i in range(len(neighbors[0])):
            suggestion = movie_piv.index[neighbors[0][i]]
            st.markdown(
                f"""
                <div style="text-align: center; font-size:20px;">
                    {suggestion}
                </div>
                """, unsafe_allow_html=True
            )
            
st.write('-------')