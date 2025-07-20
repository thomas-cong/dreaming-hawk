import regex as re
import numpy as np
from sentence_transformers import SentenceTransformer
# Pre-compiled regex patterns for efficiency
# Words are sequences of alphanumerics; we purposefully **exclude** apostrophes / hyphens so
# that ``rock'n'roll`` -> ``rock``, ``n``, ``roll`` and "don't" -> "dont".
_WORD_PATTERN = re.compile(r"[\p{L}\p{N}]+", re.UNICODE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_model = SentenceTransformer('all-MiniLM-L6-v2')
def _clean_tokens(tokens):
    """Helper – remove apostrophes / hyphens that may slip through and drop empties."""
    return [re.sub(r"[-']", "", t) for t in tokens if t]

def split_text(text: str, mode: str = 'words'):
    """Tokenise *text* according to *mode* while ensuring the result is consistent.

    Specifically, the list of word tokens returned from

    >>> [split_text(s, 'words') for s in split_text(text, 'sentences')]

    is guaranteed to be identical to ``split_text(text, 'words')`` for the same
    *text*.
    """
    text = text.lower()
    if mode == 'words':
        return _clean_tokens(_WORD_PATTERN.findall(text))
    elif mode == 'sentences':
        sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        return sentences
    elif mode == 'paragraphs':
        return [p.strip() for p in _PARAGRAPH_SPLIT_PATTERN.split(text) if p.strip()]
    else:
        raise ValueError("Mode must be 'words' or 'sentences' or 'paragraphs'")
def parse_text(file_path, mode: str = 'words'):
    """
    Read a text file and delegate tokenisation to ``split_text`` so that the
    resulting tokens are **identical** to calling ``split_text(f.read(), mode)``.

    Parameters
    ----------
    file_path : str
        Path to the text file on disk.
    mode : {'words', 'sentences', 'paragraphs'}
        Tokenisation mode. See ``split_text`` for details.

    Returns
    -------
    list[str]
        Tokens produced by ``split_text`` for the requested mode.
    """
    with open(file_path, 'r') as f:
        text = f.read()
    return split_text(text, mode=mode)

def encode_text(text):
    return _model.encode(text)
def cosine_text_similarity(text1, text2):
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
if __name__ == "__main__":
    canon = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/Letter.txt", mode='paragraphs')
    for chunk in canon:
        print("\n")
        print(chunk) 
    