import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open('D:\project\Movie success prediction\movie_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Preprocessing function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
# Preprocessing function
def preprocess_input(user_input):
    # Define all necessary columns including the additional ones
    all_columns = ['Rank', 'Title', 'Description', 'Director', 'Actors', 'Year', 
                   'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 
                   'Metascore', 'Votes_transformed', 'Action', 'Adventure', 
                   'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 
                   'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 
                   'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 
                   'Western']
    
    # Add missing columns to user input
    user_df = pd.DataFrame([user_input])
    user_df = user_df.reindex(columns=all_columns, fill_value=0)  # Fill missing columns with 0
    
    # Preprocess numerical columns if needed (e.g., scaling)
    scaler = StandardScaler()  # Use the same scaler used during model training
    num_cols = ['Year', 'Runtime (Minutes)', 'Rating', 'Revenue (Millions)', 
                'Metascore', 'Votes_transformed']
    user_df[num_cols] = scaler.fit_transform(user_df[num_cols])  # Scale numerical columns
    
     # Preprocess description using TF-IDF vectorization
    description_vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
    description_features = description_vectorizer.fit_transform(user_df['Description'])
    description_df = pd.DataFrame(description_features.toarray(), columns=description_vectorizer.get_feature_names_out())
    # Tokenize actors' names and one-hot encode them
    mlb = MultiLabelBinarizer()
    actors_df = pd.DataFrame(mlb.fit_transform([user_input['Actors'].split(',')]), columns=mlb.classes_)
    # One-hot encode categorical features like director's name
    director_encoder = OneHotEncoder(handle_unknown='ignore')  # Ignore unknown categories
    director_features = director_encoder.fit_transform(user_df[['Director']])
    director_df = pd.DataFrame(director_features.toarray(), columns=director_encoder.get_feature_names_out())
    
    # Combine numerical and categorical features
    user_preprocessed = pd.concat([user_df.drop(['Description', 'Director','Actors'], axis=1),description_df,actors_df, director_df], axis=1)
    
    return user_preprocessed



# Function to make predictions
def make_prediction(model, user_input):
    user_preprocessed = preprocess_input(user_input)
    prediction = model.predict(user_preprocessed)
    return prediction

# Streamlit UI
st.title('Movie Success Prediction')

# Sidebar for user input
st.sidebar.header('Enter Movie Details')

# User input fields
description = st.sidebar.text_area('Description', 'Enter movie description here')
actors = st.sidebar.text_input('Actors', 'Enter actors separated by commas')
director = st.sidebar.text_input('Director', 'Enter director name')
year = st.sidebar.number_input('Year', min_value=1900, max_value=2100, value=2016)
runtime = st.sidebar.number_input('Runtime (Minutes)', min_value=0, value=117)
rating = st.sidebar.number_input('Rating', min_value=0.0, max_value=10.0, value=7.3, step=0.1)
revenue = st.sidebar.number_input('Revenue (Millions)', min_value=0.0, value=138.12)
votes_transformed = st.sidebar.number_input('Votes Transformed', min_value=0.0, value=500.0)

# Genre checkboxes
st.sidebar.header('Genre')
genres = {
    'Action': st.sidebar.checkbox('Action'),
    'Adventure': st.sidebar.checkbox('Adventure'),
    'Animation': st.sidebar.checkbox('Animation'),
    'Biography': st.sidebar.checkbox('Biography'),
    'Comedy': st.sidebar.checkbox('Comedy'),
    'Crime': st.sidebar.checkbox('Crime'),
    'Drama': st.sidebar.checkbox('Drama', value=True),  # Set Drama as default
    'Family': st.sidebar.checkbox('Family'),
    'Fantasy': st.sidebar.checkbox('Fantasy'),
    'History': st.sidebar.checkbox('History'),
    'Horror': st.sidebar.checkbox('Horror'),
    'Music': st.sidebar.checkbox('Music'),
    'Mystery': st.sidebar.checkbox('Mystery'),
    'Romance': st.sidebar.checkbox('Romance'),
    'Sci-Fi': st.sidebar.checkbox('Sci-Fi'),
    'Sport': st.sidebar.checkbox('Sport'),
    'Thriller': st.sidebar.checkbox('Thriller', value=True),  # Set Thriller as default
    'War': st.sidebar.checkbox('War'),
    'Western': st.sidebar.checkbox('Western')
}

# Combine genre checkboxes into a single list
selected_genres = [genre for genre, selected in genres.items() if selected]

# Preprocess user input and make prediction
user_input = {
    'Description': description,
    'Actors': actors,
    'Director': director,
    'Year': year,
    'Runtime (Minutes)': runtime,
    'Rating': rating,
    'Revenue (Millions)': revenue,
    'Votes_transformed': votes_transformed
}
for genre in selected_genres:
    user_input[genre] = 1  # Set default value of 1 for selected genres

if st.sidebar.button('Predict'):
    prediction = make_prediction(best_model, user_input)
    if prediction == 1:
        st.write('The movie is predicted to be successful.')
    else:
        st.write('The movie is predicted to be unsuccessful.')
