import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from googleapiclient.discovery import build

df = pd.read_csv('model_ready.csv')

# Define features and target
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]
y = df['track_genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# Define YouTube API key
YOUTUBE_API_KEY = 'AIzaSyA2kAHMSBFdAhtj8kFS_MneIcbp1G8_v8k'

# Function to get YouTube link
def get_youtube_link(song_name):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=song_name, part='snippet', maxResults=1, type='video')
    response = request.execute()
    if response['items']:
        video_id = response['items'][0]['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        return video_url
    else:
        return 'No video found'

# Function to recommend songs
def recommend_songs(genre, artist, df, model, scaler, features, n_recommendations=5):
    genre = genre.strip()
    artist = artist.strip()

    # Filter the dataset based on genre and artist
    filtered_df = df[(df['track_genre'] == genre) & (df['main_artist'] == artist)]

    # Check if there are enough songs to recommend
    if len(filtered_df) < n_recommendations:
        return f"Not enough songs found for genre '{genre}' and artist '{artist}'."

    # Select features for recommendation
    input_features = filtered_df[features]
    input_features_scaled = scaler.transform(input_features)

    # Predict the genre (to ensure matching with provided genre)
    predicted_genres = model.predict(input_features_scaled)

    # Filter the recommendations based on predicted genres
    recommendations = filtered_df[predicted_genres == genre][['track_name', 'main_artist']].head(n_recommendations)

    # Get YouTube links for each recommended song
    recommendations['youtube_link'] = recommendations['track_name'].apply(get_youtube_link)

    return recommendations

# Streamlit interface
st.title("Song Recommendation")

# Load the dataset
df = pd.read_csv('spotifydataset.csv')
df.dropna(inplace=True)
df['explicit'] = df['explicit'].astype(int)
df['track_genre'] = df['track_genre'].astype('category')
df.drop_duplicates(inplace=True)
df['duration_min'] = df['duration_ms'] / 60000
df['main_artist'] = df['artists'].apply(lambda x: x.split(';')[0])

# User input for genre and artist
selected_genre = st.selectbox("Choose the Genre", df['track_genre'].unique())
selected_artist = st.selectbox("Choose the Artist", df['main_artist'].unique())

# Display selected genre and artist
st.write("Selected genre is:", selected_genre)
st.write("Selected artist is:", selected_artist)

# Recommend songs and display results
if st.button("Recommend Songs"):
    recommendations = recommend_songs(selected_genre, selected_artist, df, model, scaler, features)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(f"Recommendations for genre '{selected_genre}' and artist '{selected_artist}':")
        for i, row in recommendations.iterrows():
            st.write(f"{i + 1}. {row['track_name']} - [YouTube Link]({row['youtube_link']})")



