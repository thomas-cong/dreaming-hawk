import re

# Pre-compiled regex patterns for efficiency
_WORD_PATTERN = re.compile(r"\b\w+\b")
# Split on one or more sentence-ending punctuation marks (., !, ?)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
def parse_text(file_path, mode = 'words'):
    with open(file_path, 'r') as f:
        text = f.read()
    if mode == 'words':
        # Use the pre-compiled pattern for faster repeated calls
        return _WORD_PATTERN.findall(text)
    elif mode == 'sentences':
        # Split on period, exclamation mark, or question mark
        sentence_list = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
        return sentence_list
    else:
        raise ValueError("Mode must be 'words' or 'sentences'")
if __name__ == "__main__":
    sentences = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/Letter.txt", mode = 'sentences')
    for sentence in sentences:
        print(sentence)
    