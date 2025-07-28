from dreaming_hawk import wordGraph


text = "Hello, my name is Thomas. I like to eat apples, bananas, oranges. Apples. Bananas. Oranges."
g = wordGraph.WordGraph(text_window_size=3)
def test_create_word_graph():    
    g.add_text(text)
    nodes = g.nodes()
    window = g.get_window()
    assert len(g.edges()) > 0
    assert len(nodes) == 12
    assert window == ["apples", "bananas", "oranges"]
def test_embedding_memo():
    memod_words = {"hello", "my", "name", "is", "thomas", "i", "like", "to", "eat", "apples", "bananas", "oranges"}
    intersection = memod_words.intersection(set(g.embedding_memo.keys()))
    assert intersection == memod_words


