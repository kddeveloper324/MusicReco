"""
Generate lightweight feature vectors for music recommendation system.
This replaces the large similarity.pkl with on-the-fly TF-IDF vectorization.

Run this script once to generate 'tfidf_model.pkl' (~2-3 MB)
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def generate_tfidf_model():
    """
    Generate and save TF-IDF vectorizer and sparse matrix.
    This is much smaller than pre-computed similarity matrix.
    """
    
    # Load data
    print("Loading data...")
    df = pickle.load(open('df.pkl', 'rb'))
    
    # Create TF-IDF vectorizer (optimized for small file size)
    print("Creating TF-IDF model...")
    tfidf = TfidfVectorizer(
        max_features=1000,  # Limit features for smaller file
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    # Fit on text data
    tfidf_matrix = tfidf.fit_transform(df['text'])
    
    # Save model (much smaller than similarity matrix!)
    print("Saving TF-IDF model...")
    model_data = {
        'tfidf': tfidf,
        'matrix': tfidf_matrix
    }
    
    with open('tfidf_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Check file sizes
    tfidf_size = os.path.getsize('tfidf_model.pkl') / (1024 * 1024)
    print(f"\n✅ TF-IDF model saved successfully!")
    print(f"   File size: {tfidf_size:.2f} MB")
    print(f"   Original similarity.pkl: 190.74 MB")
    print(f"   Reduction: {100 - (tfidf_size/190.74)*100:.1f}%")


if __name__ == "__main__":
    generate_tfidf_model()
