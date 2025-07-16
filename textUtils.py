import re
import numpy as np
from sentence_transformers import SentenceTransformer
# Pre-compiled regex patterns for efficiency
_WORD_PATTERN = re.compile(r"\b\w+\b")
# Split on one or more sentence-ending punctuation marks (., !, ?)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_model = SentenceTransformer('all-MiniLM-L6-v2')
def parse_text(file_path, mode = 'words'):
    '''
    Parses a text file into a list of words or sentences.
    '''
    with open(file_path, 'r') as f:
        # Read the file and convert to lowercase
        text = f.read().lower()

    if mode == 'words':
        # Find all word sequences, which are now lowercase.
        # This effectively strips punctuation that separates words.
        return _WORD_PATTERN.findall(text)
    elif mode == 'sentences':
        # Split the text into sentences.
        sentences = _SENTENCE_SPLIT_PATTERN.split(text)
        # Clean each sentence by removing punctuation.
        cleaned_sentences = []
        for s in sentences:
            if s.strip():
                # Remove any character that is not a word character or whitespace.
                cleaned_s = re.sub(r'[^\w\s]', '', s)
                if cleaned_s.strip():
                    cleaned_sentences.append(cleaned_s.strip())
        return cleaned_sentences
    else:
        raise ValueError("Mode must be 'words' or 'sentences'")
def encode_text(text):
    return _model.encode(text)
def cosine_text_similarity(text1, text2):
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
if __name__ == "__main__":
    canon = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/FullSherlockHolmes.txt", mode='words')
    for window in sliding_window(canon, window_size=10, stride=3):
        print(window)
    