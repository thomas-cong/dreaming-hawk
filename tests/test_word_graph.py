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
def test_in_out_edges():
    result = g.in_out_edges("apples")
    assert result["in"][0] == ("to", "apples")
    assert result["out"][0] == ("apples", "bananas")
    
    

def test_embedding_memo():
    memod_words = {
        "hello",
        "my",
        "name",
        "is",
        "thomas",
        "i",
        "like",
        "to",
        "eat",
        "apples",
        "bananas",
        "oranges",
    }
    intersection = memod_words.intersection(set(g.embedding_memo.keys()))
    assert intersection == memod_words


def test_decrement_node():
    g.minus_word_node("apples")
    g.minus_word_node("apples")
    assert g.get_word_node_data("apples") == None


def test_add_temporal_edge():
    g.add_temporal_edge("hello", "thomas")
    assert g.has_edge("hello", "thomas")


def test_tick():
    time = g.time
    g.tick()
    assert g.time == time + 1
    assert g.has_edge("hello", "thomas")
    g.tick()
    assert g.time == time + 2
    assert g.has_edge("hello", "thomas")


def test_add_sentence():
    sentence = "The blue bird flies and"
    g.add_text(sentence)
    assert g.get_sentence() == ["the", "blue", "bird", "flies", "and"]
    g.add_text("sings.")
    assert g.get_sentence() == []
    g.add_text("Die.")
    assert g.get_sentence() == ["die"]
    g.add_text("song.")
    assert g.get_sentence() == []
    assert g.has_edge("sings", "song")
