import textUtils
import numpy as np
def test_encode_batch_equivalence():
    """
    Test that encoding a batch of words is equivalent to encoding each word
    individually.
    """
    words = ["apple", "banana", "orange"]
    single = {w: textUtils.encode_text(w) for w in words}
    batch = textUtils.encode_batch(words)
    for w in words:
        assert np.allclose(single[w], batch[w], atol=1e-6)
def test_split_text_modalities():
    """
    Test that split_text modalities are consistent.
    """
    text = "This is a test. This is only a test."
    words = textUtils.split_text(text, mode="words")
    sentences = textUtils.split_text(text, mode="sentences")
    paragraphs = textUtils.split_text(text, mode="paragraphs")
    assert len(words) == 9
    assert len(sentences) == 2
    assert len(paragraphs) == 1
def test_text_workhorse1():
    text1 = "Hello, my name is Thomas."
    text1_info = textUtils.extract_all_text_info(text1)
    assert text1_info["words"] == ["hello", "my", "name", "is", "thomas"]
    assert text1_info["paragraphs"] == ["hello, my name is thomas."]
    assert text1_info["sentences"] == ["hello, my name is thomas"]
    assert text1_info["sentence_ends"] == [23]
    assert text1_info["word_starts"] == [0, 7, 10, 15, 18]
    assert text1_info["sentence_ending_words"] == [4]
def test_text_workhorse2():
    text2 = "Died."
    text2_info = textUtils.extract_all_text_info(text2)
    assert text2_info["words"] == ["died"]
    assert text2_info["paragraphs"] == ["died."]
    assert text2_info["sentences"] == ["died"]
    assert text2_info["sentence_ends"] == [3]
    assert text2_info["word_starts"] == [0]
    assert text2_info["sentence_ending_words"] == [0]
    
