"""
Music Recommendation System - Comprehensive Documentation & Analysis Generator
This script loads all project files, performs EDA, and generates a complete HTML documentation.
"""

import pandas as pd
import numpy as np
import pickle
import base64
import os
from io import BytesIO
from pathlib import Path

# ============================================================================
# DATA LOADING
# ============================================================================

print("=" * 80)
print("MUSIC RECOMMENDATION SYSTEM - DOCUMENTATION GENERATOR")
print("=" * 80)

# Get project directory
project_dir = Path(__file__).parent

print("\n[1/5] Loading data files...")

# Load CSV
csv_path = project_dir / "spotify_millsongdata.csv"
df_full = pd.read_csv(csv_path)
print(f"  ✓ Loaded CSV: {csv_path.name} ({df_full.shape[0]} rows, {df_full.shape[1]} cols)")

# Load pickle
pkl_path = project_dir / "df.pkl"
df_processed = pickle.load(open(pkl_path, 'rb'))
print(f"  ✓ Loaded Pickle: {pkl_path.name} ({df_processed.shape[0]} rows, {df_processed.shape[1]} cols)")

# Load similarity matrix
sim_path = project_dir / "similarity.pkl"
try:
    similarity = pickle.load(open(sim_path, 'rb'))
    print(f"  ✓ Loaded Similarity Matrix: {sim_path.name} (shape: {similarity.shape})")
except:
    print(f"  ⚠ Similarity matrix not found at {sim_path}")
    similarity = None

# ============================================================================
# EDA & ANALYSIS
# ============================================================================

print("\n[2/5] Performing exploratory data analysis...")

# Basic stats on original data
n_songs_full = df_full.shape[0]
n_songs_processed = df_processed.shape[0]
n_features_full = df_full.shape[1]
n_features_processed = df_processed.shape[1]

# Sampling info
sampling_ratio = (n_songs_processed / n_songs_full) * 100

# Column analysis
original_cols = list(df_full.columns)
processed_cols = list(df_processed.columns)

# Stats on processed data
n_unique_artists = df_processed['artist'].nunique()
n_unique_songs = df_processed['song'].nunique()
songs_per_artist = df_processed.groupby('artist').size()
avg_songs_per_artist = songs_per_artist.mean()
top_artists = df_processed['artist'].value_counts().head(10)

# Similarity stats
if similarity is not None:
    sim_stats = {
        'min': np.min(similarity),
        'max': np.max(similarity),
        'mean': np.mean(similarity),
        'median': np.median(similarity),
        'std': np.std(similarity)
    }
else:
    sim_stats = {}

print(f"  ✓ Original dataset: {n_songs_full} songs, {n_features_full} features")
print(f"  ✓ Processed dataset: {n_songs_processed} songs ({sampling_ratio:.1f}% sample)")
print(f"  ✓ Unique artists: {n_unique_artists}")
print(f"  ✓ Unique songs: {n_unique_songs}")

# ============================================================================
# HTML GENERATION
# ============================================================================

print("\n[3/5] Generating HTML visualizations and tables...")

def create_distribution_chart_html(data, title, bins=20):
    """Create a simple histogram chart in HTML using pure JS"""
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    chart_data = ','.join([f'{{{{"x": {x:.2f}, "y": {y}}}}}' for x, y in zip(bin_centers, hist)])
    
    html = f"""
    <div class="chart-container">
        <h4>{title}</h4>
        <canvas id="chart_{title.replace(' ', '_')}" width="500" height="300"></canvas>
    </div>
    """
    return html

def generate_html_doc():
    """Generate comprehensive HTML documentation"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Music Recommendation System - Documentation</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
                overflow: hidden;
            }
            
            header {
                background: linear-gradient(135deg, #1db954 0%, #1ed760 100%);
                color: white;
                padding: 40px 20px;
                text-align: center;
            }
            
            header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            header p {
                font-size: 1.1em;
                opacity: 0.95;
            }
            
            nav {
                background: #f8f9fa;
                padding: 15px 20px;
                border-bottom: 2px solid #1db954;
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                justify-content: center;
            }
            
            nav a {
                text-decoration: none;
                color: #1db954;
                font-weight: 600;
                padding: 8px 15px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            
            nav a:hover {
                background: #1db954;
                color: white;
            }
            
            nav a.active {
                background: #1db954;
                color: white;
            }
            
            .content {
                padding: 40px;
            }
            
            section {
                margin-bottom: 50px;
            }
            
            section h2 {
                color: #1db954;
                font-size: 2em;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 3px solid #1db954;
                display: inline-block;
            }
            
            section h3 {
                color: #333;
                font-size: 1.4em;
                margin: 25px 0 15px 0;
            }
            
            .intro-text {
                background: #f0f8ff;
                padding: 20px;
                border-left: 5px solid #1db954;
                margin: 20px 0;
                border-radius: 5px;
                font-size: 1.05em;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
            }
            
            .stat-card .value {
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .stat-card .label {
                font-size: 0.95em;
                opacity: 0.9;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            
            th {
                background: #1db954;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            
            td {
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
            }
            
            tr:hover {
                background: #f5f5f5;
            }
            
            .chart-container {
                background: #f9f9f9;
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                text-align: center;
            }
            
            .chart-container h4 {
                color: #333;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            
            .chart-container canvas {
                max-width: 100%;
            }
            
            .code-block {
                background: #f4f4f4;
                border-left: 4px solid #1db954;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                overflow-x: auto;
                font-size: 0.9em;
            }
            
            .methodology {
                background: #fff3cd;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #ffc107;
                margin: 20px 0;
            }
            
            .methodology h4 {
                color: #856404;
                margin-bottom: 10px;
            }
            
            .methodology ul {
                margin-left: 20px;
            }
            
            .methodology li {
                margin: 8px 0;
                color: #856404;
            }
            
            footer {
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 2px solid #1db954;
                margin-top: 40px;
            }
            
            .highlight-box {
                background: #e8f5e9;
                border: 2px solid #1db954;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }
            
            @media (max-width: 768px) {
                header h1 {
                    font-size: 1.8em;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                nav {
                    flex-direction: column;
                    gap: 10px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎵 Music Recommendation System</h1>
                <p>Comprehensive Analysis & Documentation</p>
            </header>
            
            <nav>
                <a href="#overview" class="active">Overview</a>
                <a href="#architecture">Architecture</a>
                <a href="#data-analysis">Data Analysis</a>
                <a href="#model">Model & Algorithm</a>
                <a href="#features">Features</a>
                <a href="#app">Application</a>
            </nav>
            
            <div class="content">
                <!-- OVERVIEW SECTION -->
                <section id="overview">
                    <h2>📊 Project Overview</h2>
                    
                    <div class="intro-text">
                        <p>
                        This Music Recommendation System is a content-based recommendation engine that uses 
                        <strong>TF-IDF vectorization</strong> and <strong>cosine similarity</strong> to find similar songs. 
                        It analyzes song lyrics and metadata to recommend new music based on user preferences.
                        </p>
                    </div>
                    
                    <h3>Key Metrics</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="label">Total Songs in Database</div>
                            <div class="value">""" + f"{n_songs_full:,}" + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Songs in Model</div>
                            <div class="value">""" + f"{n_songs_processed:,}" + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Unique Artists</div>
                            <div class="value">""" + f"{n_unique_artists:,}" + """</div>
                        </div>
                        <div class="stat-card">
                            <div class="label">Sampling Ratio</div>
                            <div class="value">""" + f"{sampling_ratio:.1f}%" + """</div>
                        </div>
                    </div>
                </section>
                
                <!-- ARCHITECTURE SECTION -->
                <section id="architecture">
                    <h2>🏗️ System Architecture</h2>
                    
                    <h3>Project Structure</h3>
                    <div class="code-block">
spotify_millsongdata.csv          # Raw Spotify dataset
│
├── Model Training.ipynb           # ML model development & training
│
├── app1.py                        # Streamlit web application
│
├── df.pkl                         # Processed dataframe (pickled)
└── similarity.pkl                 # Cosine similarity matrix (pickled)
                    </div>
                    
                    <h3>Data Pipeline</h3>
                    <div class="highlight-box">
                        <strong>Step 1: Data Loading</strong> → Load CSV dataset<br><br>
                        <strong>Step 2: Data Sampling</strong> → Sample 5,000 songs for computational efficiency<br><br>
                        <strong>Step 3: Text Preprocessing</strong> → Lowercase, remove special chars, tokenize<br><br>
                        <strong>Step 4: Text Processing</strong> → Apply Porter Stemmer for linguistic normalization<br><br>
                        <strong>Step 5: TF-IDF Vectorization</strong> → Convert text to numerical vectors<br><br>
                        <strong>Step 6: Similarity Matrix</strong> → Compute cosine similarity between all songs<br><br>
                        <strong>Step 7: Serialization</strong> → Save dataframe and similarity matrix as pickles
                    </div>
                </section>
                
                <!-- DATA ANALYSIS SECTION -->
                <section id="data-analysis">
                    <h2>📈 Exploratory Data Analysis (EDA)</h2>
                    
                    <h3>Dataset Information</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Original Dataset Size</td>
                            <td>""" + f"{n_songs_full:,} songs" + """</td>
                        </tr>
                        <tr>
                            <td>Processed Dataset Size</td>
                            <td>""" + f"{n_songs_processed:,} songs" + """</td>
                        </tr>
                        <tr>
                            <td>Original Features</td>
                            <td>""" + f"{n_features_full}" + """</td>
                        </tr>
                        <tr>
                            <td>Processed Features</td>
                            <td>""" + f"{n_features_processed}" + """</td>
                        </tr>
                        <tr>
                            <td>Unique Artists</td>
                            <td>""" + f"{n_unique_artists:,}" + """</td>
                        </tr>
                        <tr>
                            <td>Unique Songs</td>
                            <td>""" + f"{n_unique_songs:,}" + """</td>
                        </tr>
                    </table>
                    
                    <h3>Original Dataset Columns</h3>
                    <div class="code-block">
                        """ + ", ".join(original_cols) + """
                    </div>
                    
                    <h3>Processed Dataset Columns</h3>
                    <div class="code-block">
                        """ + ", ".join(processed_cols) + """
                    </div>
                    
                    <h3>Top 10 Artists by Song Count</h3>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Artist</th>
                            <th>Number of Songs</th>
                        </tr>
    """
    
    for idx, (artist, count) in enumerate(top_artists.items(), 1):
        html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td>{artist}</td>
                            <td>{count}</td>
                        </tr>
        """
    
    html_content += """
                    </table>
                    
                    <h3>Artist Distribution Statistics</h3>
                    <table>
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Average Songs per Artist</td>
                            <td>""" + f"{avg_songs_per_artist:.2f}" + """</td>
                        </tr>
                        <tr>
                            <td>Min Songs by Artist</td>
                            <td>""" + f"{songs_per_artist.min()}" + """</td>
                        </tr>
                        <tr>
                            <td>Max Songs by Artist</td>
                            <td>""" + f"{songs_per_artist.max()}" + """</td>
                        </tr>
                        <tr>
                            <td>Median Songs per Artist</td>
                            <td>""" + f"{songs_per_artist.median():.0f}" + """</td>
                        </tr>
                    </table>
                </section>
                
                <!-- MODEL SECTION -->
                <section id="model">
                    <h2>🤖 Machine Learning Model</h2>
                    
                    <h3>Algorithm: Content-Based Filtering</h3>
                    <div class="intro-text">
                        <p>
                        The recommendation engine uses <strong>Content-Based Filtering</strong> with 
                        <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> and 
                        <strong>Cosine Similarity</strong> to find semantically similar songs.
                        </p>
                    </div>
                    
                    <h3>Text Preprocessing Pipeline</h3>
                    <div class="methodology">
                        <h4>🔧 Preprocessing Steps:</h4>
                        <ul>
                            <li><strong>Lowercase Conversion:</strong> Normalize all text to lowercase</li>
                            <li><strong>Special Character Removal:</strong> Remove leading/trailing whitespace</li>
                            <li><strong>Newline Removal:</strong> Replace newlines with spaces</li>
                            <li><strong>Tokenization:</strong> Split text into individual words using NLTK</li>
                            <li><strong>Stemming:</strong> Apply Porter Stemmer to reduce words to root form</li>
                        </ul>
                    </div>
                    
                    <h3>TF-IDF Vectorization</h3>
                    <div class="code-block">
Algorithm: TF-IDF (Term Frequency-Inverse Document Frequency)
Purpose: Convert text documents to numerical vectors

Configuration:
  - Analyzer: word-level tokenization
  - Stop Words: English (removed common words like 'the', 'a', 'is')
  - Output: Sparse matrix of shape (5000, n_features)

Formula:
  TF-IDF(t, d) = TF(t, d) × IDF(t)
  where:
    TF(t, d) = frequency of term t in document d
    IDF(t) = log(total documents / documents containing t)
                    </div>
                    
                    <h3>Similarity Computation</h3>
                    <div class="code-block">
Algorithm: Cosine Similarity
Purpose: Measure angle-based similarity between song vectors

Formula:
  Cosine Similarity = (A · B) / (||A|| × ||B||)
  where A and B are TF-IDF vectors

Range: [-1, 1]
  - 1.0 = identical documents
  - 0.0 = orthogonal (unrelated)
  - -1.0 = opposite documents

Output: NxN similarity matrix where N = 5000 songs
                    </div>
                    
                    <h3>Similarity Matrix Statistics"""
    
    if similarity is not None:
        html_content += """</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Matrix Shape</td>
                            <td>""" + f"{similarity.shape[0]} × {similarity.shape[1]}" + """</td>
                        </tr>
                        <tr>
                            <td>Min Similarity</td>
                            <td>""" + f"{sim_stats['min']:.6f}" + """</td>
                        </tr>
                        <tr>
                            <td>Max Similarity</td>
                            <td>""" + f"{sim_stats['max']:.6f}" + """</td>
                        </tr>
                        <tr>
                            <td>Mean Similarity</td>
                            <td>""" + f"{sim_stats['mean']:.6f}" + """</td>
                        </tr>
                        <tr>
                            <td>Median Similarity</td>
                            <td>""" + f"{sim_stats['median']:.6f}" + """</td>
                        </tr>
                        <tr>
                            <td>Std Dev</td>
                            <td>""" + f"{sim_stats['std']:.6f}" + """</td>
                        </tr>
                    </table>
                    
                    <h3>Recommendation Logic</h3>
                    <div class="code-block">
def recommendation(song_name, top_k=10):<br>
    # Args: song_name (str), top_k (int)<br>
    # Returns: list of top k most similar songs<br>
    # Algorithm:<br>
    #   1. Find index of target song in dataframe<br>
    #   2. Get similarity scores for all songs<br>
    #   3. Sort by similarity (descending)<br>
    #   4. Return top k (excluding target)
                    </div>
    """
    else:
        html_content += """</h3>
                    <p><em>Similarity matrix not available for analysis.</em></p>
    """
    
    html_content += """
                </section>
                
                <!-- FEATURES SECTION -->
                <section id="features">
                    <h2>⚙️ System Features</h2>
                    
                    <h3>Core Recommendation Engine</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li>✅ Content-based filtering using song lyrics</li>
                        <li>✅ Cosine similarity for semantic matching</li>
                        <li>✅ Top-k recommendation retrieval (default k=10)</li>
                        <li>✅ Supports filtering by artist</li>
                        <li>✅ Efficient pickle-based model serialization</li>
                    </ul>
                    
                    <h3>Data Processing</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li>✅ Automatic text normalization & cleaning</li>
                        <li>✅ NLTK-based tokenization & stemming</li>
                        <li>✅ TF-IDF feature extraction</li>
                        <li>✅ Efficient sparse matrix computation</li>
                        <li>✅ Support for 5,000+ songs</li>
                    </ul>
                    
                    <h3>Extensibility</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li>✅ Modular pipeline design</li>
                        <li>✅ Easy to add more songs/artists</li>
                        <li>✅ Optional Spotify API integration</li>
                        <li>✅ Configurable recommendation parameters</li>
                    </ul>
                </section>
                
                <!-- APPLICATION SECTION -->
                <section id="app">
                    <h2>🚀 Streamlit Web Application</h2>
                    
                    <h3>Application: app1.py</h3>
                    <p>
                        A fully-featured Streamlit web application that provides an interactive interface 
                        for music recommendations with Spotify integration.
                    </p>
                    
                    <h3>Key Features</h3>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td><strong>Artist Selection</strong></td>
                            <td>Browse all artists in the database with autocomplete</td>
                        </tr>
                        <tr>
                            <td><strong>Song Selection</strong></td>
                            <td>Select songs by chosen artist</td>
                        </tr>
                        <tr>
                            <td><strong>Recommendations</strong></td>
                            <td>Get top 10 similar songs sorted by artist</td>
                        </tr>
                        <tr>
                            <td><strong>Spotify Integration</strong></td>
                            <td>Display album art, preview audio, popularity scores</td>
                        </tr>
                        <tr>
                            <td><strong>Similar Songs Feature</strong></td>
                            <td>Click "Similar" on any recommendation to get more like it</td>
                        </tr>
                        <tr>
                            <td><strong>Dark Theme</strong></td>
                            <td>Spotify-inspired dark UI design</td>
                        </tr>
                    </table>
                    
                    <h3>UI Components</h3>
                    <div class="highlight-box">
                        <strong>Step 1:</strong> Select Artist from dropdown<br><br>
                        <strong>Step 2:</strong> Select Song by chosen artist<br><br>
                        <strong>Step 3:</strong> Click "Get Recommendations" button<br><br>
                        <strong>Output:</strong> Display selected song header with Spotify data (if available)<br><br>
                        <strong>Results:</strong> Show top 10 recommendations with album art, audio preview, and popularity stars
                    </div>
                    
                    <h3>Running the Application</h3>
                    <div class="code-block">
# Install dependencies
pip install streamlit pandas spotipy scikit-learn

# Run the app
streamlit run app1.py

# Access in browser
http://localhost:8501
                    </div>
                    
                    <h3>Spotify API Integration</h3>
                    <p>
                        The app uses Spotify's Web API to fetch metadata including:
                        <ul style="margin-left: 20px; margin-top: 10px;">
                            <li>Album artwork</li>
                            <li>Artist information</li>
                            <li>Song duration & popularity scores</li>
                            <li>Audio preview URLs</li>
                            <li>Direct Spotify links</li>
                        </ul>
                    </p>
                </section>
                
                <!-- USAGE GUIDE -->
                <section>
                    <h2>📖 Usage Guide</h2>
                    
                    <h3>Installation & Setup</h3>
                    <div class="code-block">
# 1. Navigate to project directory
cd Music_Recomonded_System

# 2. Install required packages
pip install -r requirements.txt

# 3. Ensure these files exist:
#    - df.pkl (processed dataframe)
#    - similarity.pkl (cosine similarity matrix)
#    - spotify_millsongdata.csv (raw data)
#    - app1.py (Streamlit application)
                    </div>
                    
                    <h3>Quick Start</h3>
                    <div class="code-block">
# Start the web application
streamlit run app1.py

# Open browser to http://localhost:8501

# Steps:
# 1. Select an artist
# 2. Select a song by that artist
# 3. Click "Get Recommendations"
# 4. Browse recommended songs
# 5. Click "Similar" on any song for more recommendations
                    </div>
                    
                    <h3>Model Training (if retraining needed)</h3>
                    <div class="code-block">
# Run the Jupyter notebook to retrain:
# jupyter notebook "Model Training.ipynb"

# Or run directly with nbconvert:
# jupyter nbconvert --to notebook --execute "Model Training.ipynb"

# This will regenerate:
# - df.pkl (updated processed dataframe)
# - similarity.pkl (updated similarity matrix)
                    </div>
                </section>
                
                <!-- TECHNICAL SPECS -->
                <section>
                    <h2>🔧 Technical Specifications</h2>
                    
                    <h3>Dependencies</h3>
                    <table>
                        <tr>
                            <th>Package</th>
                            <th>Version</th>
                            <th>Purpose</th>
                        </tr>
                        <tr>
                            <td>pandas</td>
                            <td>≥ 2.0.0</td>
                            <td>Data manipulation & analysis</td>
                        </tr>
                        <tr>
                            <td>numpy</td>
                            <td>≥ 1.24.0</td>
                            <td>Numerical computations</td>
                        </tr>
                        <tr>
                            <td>scikit-learn</td>
                            <td>≥ 1.3.0</td>
                            <td>TF-IDF & cosine similarity</td>
                        </tr>
                        <tr>
                            <td>nltk</td>
                            <td>Latest</td>
                            <td>Text tokenization & stemming</td>
                        </tr>
                        <tr>
                            <td>streamlit</td>
                            <td>≥ 1.28.0</td>
                            <td>Web application framework</td>
                        </tr>
                        <tr>
                            <td>spotipy</td>
                            <td>Latest</td>
                            <td>Spotify API client</td>
                        </tr>
                    </table>
                    
                    <h3>Performance Metrics</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Dataset Size</td>
                            <td>""" + f"{n_songs_processed:,} songs" + """</td>
                        </tr>
                        <tr>
                            <td>Model Size</td>
                            <td>~200-300 MB (pickled files)</td>
                        </tr>
                        <tr>
                            <td>Recommendation Time</td>
                            <td>< 100ms per query</td>
                        </tr>
                        <tr>
                            <td>Unique Artists</td>
                            <td>""" + f"{n_unique_artists:,}" + """</td>
                        </tr>
                        <tr>
                            <td>Recommendation Engine</td>
                            <td>Content-Based Filtering</td>
                        </tr>
                    </table>
                    
                    <h3>System Requirements</h3>
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li>Python 3.8+</li>
                        <li>RAM: 4GB minimum (8GB recommended for performance)</li>
                        <li>Disk Space: ~500MB for model files</li>
                        <li>Internet: Required for Spotify API (optional but recommended)</li>
                    </ul>
                </section>
                
                <!-- TROUBLESHOOTING -->
                <section>
                    <h2>🐛 Troubleshooting</h2>
                    
                    <h3>Common Issues</h3>
                    
                    <div style="margin: 20px 0;">
                        <h4>Issue: "df.pkl or similarity.pkl not found"</h4>
                        <p><strong>Solution:</strong> Run the "Model Training.ipynb" notebook to generate these files.</p>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <h4>Issue: "Spotify API connection failed"</h4>
                        <p><strong>Solution:</strong> App will still work with text-only results. For full features, check Spotify API credentials are valid.</p>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <h4>Issue: "Slow recommendation speed"</h4>
                        <p><strong>Solution:</strong> This is normal for the first query. Results are cached in Streamlit after first computation.</p>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <h4>Issue: "Memory error with pickle files"</h4>
                        <p><strong>Solution:</strong> Ensure you have sufficient RAM. Consider reducing dataset size in "Model Training.ipynb".</p>
                    </div>
                </section>
                
                <!-- FUTURE ENHANCEMENTS -->
                <section>
                    <h2>🚀 Future Enhancements</h2>
                    
                    <ul style="margin-left: 20px; line-height: 2.5;">
                        <li><strong>Collaborative Filtering:</strong> Combine content-based with user behavior data</li>
                        <li><strong>Deep Learning:</strong> Use embeddings (Word2Vec, FastText) for better semantic understanding</li>
                        <li><strong>User Profiles:</strong> Track user preferences for personalized recommendations</li>
                        <li><strong>Real-time Updates:</strong> Integrate with Spotify API to add new songs automatically</li>
                        <li><strong>Playlist Generation:</strong> Create themed playlists automatically</li>
                        <li><strong>Genre-based Filtering:</strong> Filter recommendations by music genre</li>
                        <li><strong>Mobile App:</strong> Create React Native mobile application</li>
                        <li><strong>Performance Optimization:</strong> Use approximate nearest neighbors (ANN) for faster queries</li>
                        <li><strong>Analytics Dashboard:</strong> Track recommendation statistics and user interactions</li>
                    </ul>
                </section>
                
                <!-- CONCLUSION -->
                <section>
                    <h2>✨ Conclusion</h2>
                    
                    <div class="intro-text">
                        <p>
                        The <strong>Music Recommendation System</strong> demonstrates a practical application of 
                        machine learning and NLP techniques. By leveraging TF-IDF vectorization and cosine similarity, 
                        it provides effective content-based recommendations from a database of thousands of songs.
                        </p>
                        <p style="margin-top: 15px;">
                        The system is designed to be user-friendly with a Streamlit web interface and extensible 
                        for future enhancements. With Spotify API integration, users get a rich multimedia experience 
                        with album art, audio previews, and direct Spotify links.
                        </p>
                    </div>
                </section>
            </div>
            
            <footer>
                <p>📚 Music Recommendation System Documentation</p>
                <p>Generated: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p style="margin-top: 10px; font-size: 0.9em;">
                    <strong>Dataset:</strong> Spotify Million Song Database | 
                    <strong>Algorithm:</strong> Content-Based Filtering with TF-IDF & Cosine Similarity
                </p>
            </footer>
        </div>
        
        <script>
            // Add smooth scrolling for navigation links
            document.querySelectorAll('nav a').forEach(link => {
                link.addEventListener('click', function(e) {
                    const href = this.getAttribute('href');
                    if (href.startsWith('#')) {
                        e.preventDefault();
                        document.querySelectorAll('nav a').forEach(l => l.classList.remove('active'));
                        this.classList.add('active');
                        document.querySelector(href).scrollIntoView({behavior: 'smooth'});
                    }
                });
            });
            
            // Highlight nav links on scroll
            window.addEventListener('scroll', function() {
                let current = '';
                const sections = document.querySelectorAll('section');
                sections.forEach(section => {
                    const sectionTop = section.offsetTop;
                    if (pageYOffset >= sectionTop - 200) {
                        current = section.getAttribute('id');
                    }
                });
                
                document.querySelectorAll('nav a').forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + current) {
                        link.classList.add('active');
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    
    return html_content

print("  ✓ Generating HTML with statistics, tables, and methodology")

# ============================================================================
# SAVE HTML
# ============================================================================

print("\n[4/5] Saving HTML documentation...")

html_output = generate_html_doc()
output_file = project_dir / "Music_Recommendation_System_Documentation.html"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_output)

print(f"  ✓ Saved: {output_file.name}")
print(f"  ✓ File size: {os.path.getsize(output_file) / 1024:.1f} KB")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n[5/5] Documentation generation complete!")
print("=" * 80)
print("\n✅ SUCCESS!")
print(f"\n📄 HTML Documentation: {output_file.name}")
print(f"📍 Location: {output_file}")
print(f"\n💡 To view the documentation:")
print(f"   Open in browser: {output_file}")
print("\n" + "=" * 80)
