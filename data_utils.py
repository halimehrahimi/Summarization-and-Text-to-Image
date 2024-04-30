import pandas as pd
import json
import matplotlib.pyplot as plt
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer


CONTRACTIONS = {
    "ain't": "is not",
    "you ain't": "you are not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'll": "he will",
    "he's": "he is",
}


def create_dataset(path) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    """
    # Load data and specify column names
    df = pd.read_csv(path, sep='\t', header=None, names=['book_id', 'freebase_id', 'book_title', 'author', 'publication_date', 'genre', 'summary'], na_filter=False)
    # Clean the dataset
    df['genre'] = df['genre'].apply(parse_genre_entry)
    df['publication_date'] = df['publication_date'].astype(str).str[:4]
    df = df.drop_duplicates(subset=['book_id', 'summary'])
    df['summary'] = df['summary'].apply(clean_summary)
    df['summary_length'] = df['summary'].str.split().str.len()
    df = df[df['summary_length'] >= 5]
    return df

def parse_genre_entry(genre_info: str) -> list:
    """
    Parse genre information from JSON format.
    """
    if not genre_info:
        return ['Unknown']
    genre_dict = json.loads(genre_info)
    return list(genre_dict.values())

def clean_summary(summary: str) -> str:
    """
    Clean and preprocess the book summary.
    """
    summary = html.unescape(summary)  # Decode HTML entities
    summary = re.sub(r'&[^\s]+;', ' ', summary)  # Remove HTML entities
    summary = remove_punctuations(summary)  # Remove punctuations
    summary = expand_contractions(summary)  # Expand contractions
    summary = remove_words(summary)  # Remove specific phrases
    return summary

def remove_punctuations(text: str) -> str:
    """
    Remove punctuations from text.
    """
    pattern = r'[^a-zA-Z0-9\'\"\. (),]'
    clean_text = re.sub(pattern, ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Remove extra whitespaces
    return clean_text.strip()

def expand_contractions(text: str) -> str:
    """
    Expand contractions in text.
    """
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

def remove_words(text: str) -> str:
    """
    Remove specific phrases or patterns from text.
    """
    pattern = r'(In chapter \d+,|Plot outline description|As described by Sherryl Connelly of the New York Daily News,|See the articles on the separate works.|This book has not yet been released.)'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def analyze_dataset(df: pd.DataFrame) -> None:
    """
    Analyze the dataset and generate summary statistics.
    """
    df_describe = df.describe()
    df_describe.to_csv('books_dataframe_summary.csv')
    # Boxplot of Summary Length
    fig = plt.figure(figsize=(6, 8))
    plt.boxplot(df['summary_length'], showfliers=False)
    plt.ylabel('Number of Words')
    plt.title('Boxplot of Summary Length')
    plt.grid(True)
    plt.show()
    fig.savefig('books_dataframe_summary.png')

def keyword_extraction(text: str) -> list:
    """
    Extract keywords using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(max_features=5, stop_words='english', use_idf=True)
    _ = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names)
    return feature_names