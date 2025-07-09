import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess raw text input."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)                        
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Tokenize and remove stopwords."""
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return filtered

def preprocess(text):
    """Full preprocessing pipeline."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return ' '.join(tokens)
