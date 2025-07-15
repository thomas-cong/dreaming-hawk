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
def sliding_window(text, window_size=3, stride=1):
    """Yields sliding windows of a given size and stride from a list."""
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers")
    
    for i in range(0, len(text) - window_size + 1, stride):
        yield text[i:i + window_size]
    
    
if __name__ == "__main__":
    canon = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/FullSherlockHolmes.txt", mode='words')
    for window in sliding_window(canon, window_size=10, stride=3):
        print(window)
    