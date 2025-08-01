import regex as re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

# Pre-compiled regex patterns for efficiency
# Words are sequences of alphanumerics; we purposefully **exclude** apostrophes / hyphens so
# that ``rock'n'roll`` -> ``rock``, ``n``, ``roll`` and "don't" -> "dont".
_WORD_PATTERN = re.compile(r"[\p{L}\p{N}]+", re.UNICODE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_model = SentenceTransformer("all-MiniLM-L6-v2")
_lemmatizer = WordNetLemmatizer()


def _clean_tokens(tokens: list[str]):
    """Helper – remove apostrophes / hyphens that may slip through and drop empties."""
    return [re.sub(r"[-']", "", t) for t in tokens if t]


def split_text(text: str, mode: str = "words"):
    """Tokenise *text* according to *mode* while ensuring the result is consistent.

    Specifically, the list of word tokens returned from

    >>> [split_text(s, 'words') for s in split_text(text, 'sentences')]

    is guaranteed to be identical to ``split_text(text, 'words')`` for the same
    *text*.
    """
    text = text.lower()
    if mode == "words":
        return _clean_tokens(_WORD_PATTERN.findall(text))
    elif mode == "sentences":
        sentences = [
            s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()
        ]
        return sentences
    elif mode == "paragraphs":
        return [p.strip() for p in _PARAGRAPH_SPLIT_PATTERN.split(text) if p.strip()]
    else:
        raise ValueError("Mode must be 'words' or 'sentences' or 'paragraphs'")


def encode_batch(words: list[str]) -> dict[str, np.ndarray]:
    vecs = _model.encode(words)
    return dict(zip(words, vecs))


def parse_text(file_path: str, mode: str = "words"):
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
    with open(file_path, "r") as f:
        text = f.read()
    return split_text(text, mode=mode)


def encode_text(text: str):
    return _model.encode(text)


def extract_all_text_info(text: str):
    """
    Workhorse text function that returns all necessary text information to build the word graph
    """
    sentences = split_text(text, mode="sentences")
    paragraphs = split_text(text, mode="paragraphs")
    words = split_text(text, mode="words")
    word_starts = [m.start() for m in re.finditer(r"\S+", text)]
    sentence_ends = [
        match.start(1) + len(match.group(1)) - 1
        for match in re.finditer(r"(\w+)\s*(?:\.\.\.|!!!|[.!?])", text)
    ]
    ending_words = []

    search_from = 0
    word_start = word_starts[search_from]
    for end in sentence_ends:
        if search_from >= len(word_starts):
            break
        while word_start <= end and search_from < len(word_starts):
            word_start = word_starts[search_from]
            search_from += 1
        ending_words.append(search_from - 1)

    result_dict = {
        "words": words,
        "paragraphs": paragraphs,
        "sentences": sentences,
        "sentence_ends": sentence_ends,
        "word_starts": word_starts,
        "sentence_ending_words": ending_words,
    }
    return result_dict


def cosine_text_similarity(text1: str, text2: str):
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def cosine_similarity(vec1: list[float], vec2: list[float]):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def lemmatize_text(text: str):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        _lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged
    ]
    return lemmatized


def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


if __name__ == "__main__":
    source = """This is a sentence... This is a second sentence.
         Is this a sentence? Sure it is!!!"""
    text_info = extract_all_text_info(source)
    words = text_info["words"]
    ending_words = text_info["sentence_ending_words"]
