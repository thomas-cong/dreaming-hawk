import textUtils
import numpy as np

def test_encode_batch_equivalence():
    '''
    Test that encoding a batch of words is equivalent to encoding each word
    individually.
    '''
    words = ["apple", "banana", "orange"]
    single = {w: textUtils.encode_text(w) for w in words}
    batch  = textUtils.encode_batch(words)
    for w in words:
        assert np.allclose(single[w], batch[w], atol=1e-6)
def test_split_text_modalities():
    '''
    Test that split_text modalities are consistent.
    '''
    text = "This is a test. This is only a test."
    words = textUtils.split_text(text, mode="words")
    sentences = textUtils.split_text(text, mode="sentences")
    paragraphs = textUtils.split_text(text, mode="paragraphs")
    assert len(words) == 9
    assert len(sentences) == 2
    assert len(paragraphs) == 1
    

    
    