
import streamlit as st
import pickle
import pandas as pd
import random
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    st.error("Spotipy not installed. Please install it with: pip install spotipy")

# --- Page Configuration (MUST be the first st command) ---
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="centered"
)

# --- Helper Function ---
def format_duration_ms(ms):
    """Converts milliseconds to M:SS format."""
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    return f"{minutes}:{seconds:02d}"

# --- New Helper Function for Stars ---
def popularity_to_stars(popularity):
    """Converts a 0-100 popularity score to a 5-star HTML string."""
    if popularity is None:
        return ""
    # Round to nearest star
    num_stars = round(popularity / 20.0)
    full_stars = "★" * num_stars
    empty_stars = "☆" * (5 - num_stars)
    # Use markdown for inline styling
    stars_html = f'<span style="color: #FFD700; font-size: 1.1rem;">{full_stars}</span><span style="color: #535353; font-size: 1.1rem;">{empty_stars}</span>'
    return stars_html

# --- Custom CSS for a "Spotify-like" Dark Theme ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* --- Main App Styling --- */
        .stApp {
            background-color: #121212; /* Dark background */
            color: #FFFFFF;
        }

        h1, h2, h3 {
            color: #FFFFFF; /* White headers */
            font-weight: bold;
        }
        
        /* --- Recommendation List Styling --- */
        /* Make song title link white */
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] a {
            color: #FFFFFF !important;
            text-decoration: none;
            font-size: 1.1rem; /* Make song title a bit bigger */
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] a:hover {
            text-decoration: underline;
        }

        /* Make caption text (artist, pop, duration) lighter */
        [data-testid="stMain"] [data-testid="stCaptionContainer"] {
            color: #b3b3b3;
            font-size: 0.9rem;
        }
        
        /* Ensure columns are vertically centered */
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        /* --- Button Styling --- */
        /* --- Main "Get Recommendations" Button --- */
        div[data-testid="stButton"][data-container-width="true"] > button {
            background-color: #1DB954; /* Spotify Green */
            color: #FFFFFF;
            border: none;
            border-radius: 25px; /* Fully rounded */
            font-weight: bold;
            font-size: 1rem;
            padding: 0.5rem 1rem;
        }
        div[data-testid="stButton"][data-container-width="true"] > button:hover {
            background-color: #1ED760;
            color: #FFFFFF;
        }
        div[data-testid="stButton"][data-container-width="true"] > button:disabled {
            background-color: #535353; /* Grey when disabled */
            color: #b3b3b3;
        }
        
        /* --- "Similar" Button Styling (in list) --- */
        div[data-testid="stButton"]:not([data-container-width="true"]) > button {
            background-color: transparent;
            color: #FFFFFF;
            border: 1px solid #FFFFFF;
            border-radius: 25px;
            font-weight: bold;
            font-size: 0.8rem; /* Smaller font */
            padding: 0.2rem 0.6rem; /* Smaller padding */
            width: 100%;
        }
        div[data-testid="stButton"]:not([data-container-width="true"]) > button:hover {
            background-color: #282828;
            border-color: #1DB954;
            color: #1DB954;
        }

        /* --- Selectbox Styling --- */
        div[data-testid="stSelectbox"] label {
            color: #FFFFFF;
            font-size: 1.1rem;
        }
        
        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: #191919;
        }
        </style>
    """, unsafe_allow_html=True)

# Inject the CSS
inject_custom_css()

# --- Spotify API Credentials ---
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "6cea1127c8b24d51940c79241014b53e")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "b32e3235eb7e4ecb881c6eebcb5de140")

# --- Setup Spotify Client (with error handling) ---
spotify_available = False
try:
    if not SPOTIPY_AVAILABLE:
        raise ImportError("Spotipy library not installed")
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    sp.search(q="test", limit=1) 
    spotify_available = True
    st.sidebar.success("✅ Connected to Spotify API")
except Exception as e:
    st.sidebar.error("❌ Spotify API connection failed.")
    st.sidebar.warning("App will show text-only results.")

# --- Load Data (with caching) ---
@st.cache_data
def load_data():
    """
    Loads the pre-processed DataFrame and TF-IDF model from pickle files.
    Uses on-the-fly similarity calculation instead of pre-computed matrix.
    """
    try:
        df = pickle.load(open('df.pkl', 'rb'))
        
        # Try to load optimized TF-IDF model first (smaller file)
        if os.path.exists('tfidf_model.pkl'):
            model_data = pickle.load(open('tfidf_model.pkl', 'rb'))
            tfidf_model = model_data['tfidf']
            tfidf_matrix = model_data['matrix']
            return df, {'tfidf': tfidf_model, 'matrix': tfidf_matrix, 'type': 'tfidf'}
        
        # Fallback to pre-computed similarity if available
        elif os.path.exists('similarity.pkl'):
            similarity = pickle.load(open('similarity.pkl', 'rb'))
            return df, {'matrix': similarity, 'type': 'precomputed'}
        
        else:
            st.error("Error: Neither 'tfidf_model.pkl' nor 'similarity.pkl' found.")
            st.write("Please generate the TF-IDF model by running: python generate_features.py")
            return None, None
            
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found - {e}")
        st.write("Please make sure 'df.pkl' and either 'tfidf_model.pkl' or 'similarity.pkl' are in the app directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None

# --- Spotify API Functions (with caching) ---
@st.cache_data
def fetch_spotify_info(song_name, artist_name):
    """
    Searches Spotify for a track and returns its info.
    Caches results to avoid repeated API calls.
    """
    if not spotify_available:
        return None
    try:
        query = f"track:{song_name} artist:{artist_name}"
        results = sp.search(q=query, type='track', limit=1)
        
        if not results['tracks']['items']:
            query_general = f"{song_name} {artist_name}"
            results = sp.search(q=query_general, type='track', limit=1)
            if not results['tracks']['items']:
                return None 

        track = results['tracks']['items'][0]
        return {
            'name': track['name'],
            'artist': ", ".join(artist['name'] for artist in track['artists']),
            'image_url': track['album']['images'][0]['url'] if track['album']['images'] else "https.placehold.co/300x300/2c2c2c/ffffff?text=No+Art",
            'preview_url': track.get('preview_url'),
            'spotify_url': track['external_urls']['spotify'],
            'duration_str': format_duration_ms(track['duration_ms']),
            'popularity': track['popularity']
        }
    except Exception as e:
        print(f"Error fetching from Spotify: {e}") 
        return None

# --- UI Helper Function (for the recommendation LIST) ---
def display_spotify_track(song_row):
    """
    Displays a recommended track in a small list format.
    """
    spotify_info = fetch_spotify_info(song_row['song'], song_row['artist'])

    col1, col2, col3, col4, col5 = st.columns([1.5, 5, 2, 1.5, 2])

    if spotify_available and spotify_info:
        with col1:
            st.image(spotify_info['image_url'], width=70)
        
        with col2:
            st.markdown(f"**[{spotify_info['name']}]({spotify_info['spotify_url']})**") 
            st.caption(f"by {spotify_info['artist']}")
            if spotify_info['preview_url']:
                st.audio(spotify_info['preview_url'])
        
        with col3:
            st.write("") 
            star_rating = popularity_to_stars(spotify_info['popularity'])
            st.markdown(star_rating, unsafe_allow_html=True)
            
        with col4:
            st.write("") 
            st.caption(f"**{spotify_info['duration_str']}**")
        
        with col5:
            st.write("") 
            st.button(
                "Similar", 
                key=f"rec_{song_row['song']}_{song_row['artist']}", 
                on_click=set_recommendation_target,
                args=(song_row['artist'], song_row['song'])
            )
    else:
        # Fallback to simple text
        with col1:
            st.write("🎵") 
        with col2:
            st.markdown(f"**{song_row['song']}**")
            st.caption(f"by {song_row['artist']}")
        with col5:
            st.write("") 
            st.button(
                "Similar", 
                key=f"rec_{song_row['song']}_{song_row['artist']}", 
                on_click=set_recommendation_target,
                args=(song_row['artist'], song_row['song'])
            )
    
    st.divider()

# --- (NEW) UI Helper Function (for the SELECTED song) ---
def display_selected_song_header(song_name, artist_name):
    """
    Displays the user-selected song in a prominent header format.
    """
    spotify_info = fetch_spotify_info(song_name, artist_name)
    
    st.header(f"Recommendations for:")
    
    if spotify_available and spotify_info:
        col1, col2 = st.columns([1, 2]) # 1:2 ratio for image:info
        
        with col1:
            st.image(spotify_info['image_url'])
        
        with col2:
            st.title(f"**{spotify_info['name']}**")
            st.subheader(f"by {spotify_info['artist']}")
            
            # Display stars and duration
            stars = popularity_to_stars(spotify_info['popularity'])
            st.markdown(stars, unsafe_allow_html=True)
            st.caption(f"Duration: {spotify_info['duration_str']}")
            
            # Show audio preview
            if spotify_info['preview_url']:
                st.audio(spotify_info['preview_url'])
            
            # Add a button to listen on Spotify
            st.link_button("Listen on Spotify", spotify_info['spotify_url'], use_container_width=True)
            
    else:
        # Fallback if Spotify fails
        st.title(f"**{song_name}**")
        st.subheader(f"by {artist_name}")

    st.divider()


# --- Load the data ---
df, model_info = load_data()

# --- Callbacks ---
def clear_song_and_recs():
    """Callback to clear song and stop recs when artist changes."""
    st.session_state.selected_song = None
    st.session_state.run_recs = False

def clear_recs():
    """Callback to stop recs when song changes (requires new button press)."""
    st.session_state.run_recs = False

def set_recommendation_target(artist, song):
    """Callback for 'Suggest Similar' button."""
    st.session_state.selected_artist = artist
    st.session_state.selected_song = song
    st.session_state.run_recs = True
    # Scroll to top (optional, requires JS)

def trigger_recommendations():
    """Callback for main 'Get Recs' button."""
    if st.session_state.selected_song:
        st.session_state.run_recs = True

# --- Initialize Session State ---
if 'selected_artist' not in st.session_state:
    st.session_state.selected_artist = None
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None
if 'run_recs' not in st.session_state:
    st.session_state.run_recs = False


# --- Main Application Logic (MODIFIED) ---
if df is not None and model_info is not None:
    
    st.title("🎵 Music Recommendation System")
    st.write("Find new music based on what you love.")
    
    st.divider()

    # --- 1. Artist Selection ---
    try:
        artists = sorted(df['artist'].unique())
        selected_artist = st.selectbox(
            "**Step 1: Select an Artist**",
            artists,
            key='selected_artist', 
            placeholder="Choose an artist...",
            on_change=clear_song_and_recs # Clear song and recs
        )

        # --- 2. Song Selection (Conditional) ---
        if st.session_state.selected_artist:
            artist_songs = sorted(df[df['artist'] == st.session_state.selected_artist]['song'].unique())
            selected_song = st.selectbox(
                "**Step 2: Select a Song**",
                artist_songs,
                key='selected_song', 
                placeholder="Find a song by that artist...",
                on_change=clear_recs # Clear old recs
            )
        
        st.write("") 
        
        # --- 3. Recommendation Button ---
        st.button(
            "Get Recommendations", 
            disabled=(not st.session_state.selected_song), 
            use_container_width=True,
            on_click=trigger_recommendations
        )
            
        # --- 4. Recommendation Logic (MODIFIED) ---
        if st.session_state.run_recs and st.session_state.selected_song:
            
            # --- (NEW) Display the selected song as a header ---
            display_selected_song_header(
                st.session_state.selected_song, 
                st.session_state.selected_artist
            )
            
            # --- (OLD) Recommendation Logic ---
            st.header("You might also like...")
            try:
                song_index_matches = df[
                    (df['artist'] == st.session_state.selected_artist) & 
                    (df['song'] == st.session_state.selected_song)
                ].index
                
                if song_index_matches.empty:
                    st.error(f"Could not find a unique entry for '{st.session_state.selected_song}' by '{st.session_state.selected_artist}'.")
                    st.session_state.run_recs = False
                else:
                    song_index = song_index_matches[0]
                    
                    # Calculate similarities based on model type
                    if model_info['type'] == 'tfidf':
                        # On-the-fly TF-IDF similarity calculation
                        tfidf_matrix = model_info['matrix']
                        query_vector = tfidf_matrix[song_index]
                        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
                        distances = sorted(list(enumerate(similarities)), reverse=True, key=lambda x: x[1])
                    else:
                        # Pre-computed similarity matrix
                        similarity_matrix = model_info['matrix']
                        distances = sorted(list(enumerate(similarity_matrix[song_index])), reverse=True, key=lambda x: x[1])

                    same_artist_recs = []
                    other_artist_recs = []

                    for i in distances[1:21]: 
                        rec_song_index = i[0]
                        rec_song_row = df.iloc[rec_song_index]
                        
                        if rec_song_row['song'] == st.session_state.selected_song:
                            continue
                            
                        if rec_song_row['artist'] == st.session_state.selected_artist:
                            same_artist_recs.append(rec_song_row)
                        else:
                            other_artist_recs.append(rec_song_row)

                    # Part 1: Main Recommendations (Other Artists)
                    if other_artist_recs:
                        for song_row in other_artist_recs[:10]:
                            display_spotify_track(song_row)
                    else:
                        st.write("No similar songs by other artists found in the top 20.")

                    st.divider()

                    # Part 2: Top 5 Similar Songs by the Same Artist
                    st.subheader(f"More by {st.session_state.selected_artist}")
                    if same_artist_recs:
                        for song_row in same_artist_recs[:5]:
                            display_spotify_track(song_row)
                    else:
                        st.write(f"No other similar songs by {st.session_state.selected_artist} found in the top 20.")
                    
            except IndexError:
                st.error(f"Could not find '{st.session_state.selected_song}' by '{st.session_state.selected_artist}' in the recommendation index.")
                st.session_state.run_recs = False
            except Exception as e:
                st.error(f"An error occurred during recommendation: {e}")
                st.session_state.run_recs = False
    
    except Exception as e:
        st.error(f"An error occurred in the main app: {e}")
else:
    st.info("Loading recommendation engine...")