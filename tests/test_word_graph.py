from dreaming_hawk import wordGraph

def test_batch_path_no_errors():
    g = wordGraph.WordGraph(text_window_size=3)
    g.add_text("apple banana apple orange")
    # embedding_memo should contain exactly the vocabulary
    assert set(g.embedding_memo.keys()) == {"apple", "banana", "orange"}
    # ensure graph edges exist
    assert g.number_of_edges() > 0