import networkx as nx
import sys, pathlib, os

from tqdm import tqdm
import textUtils
import json
import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class NodeEncoder(json.JSONEncoder):
    """
    JSON encoder for WordNodeData and LemmaNodeData objects.
    """

    def default(self, obj):
        if isinstance(obj, WordNodeData):
            return obj.to_dict()
        elif isinstance(obj, LemmaNodeData):
            return obj.to_dict()
        # Convert NumPy scalar types to native Python types for JSON serialization
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class WordNodeData:
    """
    Data associated with a word node.
    """

    def __init__(self, word, value: int):
        self.word = word
        self.value = value
        self.lemmatized = textUtils.lemmatize_text(word)

    def set_value(self, value: int):
        self.value = value

    def get_value(self):
        return self.value

    def to_dict(self):
        return {"word": self.word, "value": self.value, "lemmatized": self.lemmatized}

    def __str__(self):
        return "WordNode(" + self.word + ", " + str(self.value) + ")"

    def __iadd__(self, other):
        self.value += other
        return self

    def __isub__(self, other):
        self.value -= other
        return self

    def __eq__(self, other):
        return (
            isinstance(other, WordNodeData)
            and self.word == other.word
            and self.value == other.value
        )

    def __hash__(self):
        # identified by hashing the lemmatized word
        return hash(self.word)


class LemmaNodeData:
    """
    Data associated with a lemma node.
    """

    def __init__(self, lemma: str):
        self.lemma = lemma

    def __str__(self):
        return "LemmaNode(" + self.lemma + ")"

    def __hash__(self):
        return hash(self.lemma)

    def to_dict(self):
        return {"lemma": self.lemma}


class LemmaGraph(nx.Graph):
    """
    Undirected graph representing the semantic connections between lemmas.
    """

    def __init__(self):
        super().__init__()

    def add_lemma_node(self, lemma: str):
        if self.has_node(lemma):
            return None
        self.add_node(lemma, data=LemmaNodeData(lemma))
        return None

    def add_lemma_edge(self, lemma1: str, lemma2: str, weight: float):
        if not self.has_node(lemma1):
            self.add_lemma_node(lemma1)
        if not self.has_node(lemma2):
            self.add_lemma_node(lemma2)
        if self.has_edge(lemma1, lemma2):
            self.update_lemma_edge(
                lemma1, lemma2, weight=max(self[lemma1][lemma2]["weight"], weight)
            )
        else:
            self.add_edge(lemma1, lemma2, weight=weight)
        return None

    def update_lemma_edge(self, lemma1: str, lemma2: str, weight: float):
        if self.has_edge(lemma1, lemma2):
            self[lemma1][lemma2]["weight"] = weight
        else:
            raise ValueError("Edge does not exist")
        return None


class WordGraph(nx.MultiDiGraph):
    """
    Multi-directional graph representing the semantic connections and temporal connections between words.
    """

    def __init__(self, text_window_size: int = 30, semantic_threshold: float = 0.5):
        super().__init__()
        self.lemma_graph = LemmaGraph()
        self.text_window_size = text_window_size
        self.semantic_threshold = semantic_threshold
        self.time = 0
        self.embedding_memo = {}
        self.sentence = []
        self.paragraph = []
        self.window = []

    def warm_up(self):
        # Warm up the nodes
        self.add_word_node("buffer")
        self.minus_word_node("buffer")
        self.add_semantic_edge("test", "exam", 1.0)
        self.remove_edge("test", "exam")
        self.minus_word_node("test")
        self.minus_word_node("exam")

    def get_window(self):
        return self.window.copy()

    def get_time(self):
        return self.time

    def get_lemma_graph(self):
        return self.lemma_graph.copy()

    def get_sentence(self):
        return self.sentence.copy()

    def get_paragraph(self):
        return self.paragraph.copy()

    def add_word_node(self, word: str) -> None:
        """
        Adds a word to the graph or increments its value if it already exists.
        """
        if self.has_node(word):
            # Access the data via the string `word`
            self.nodes[word]["data"] += 1
        else:
            # Use the string `word` as the node and store WordNode in an attribute
            node_data = WordNodeData(word, 1)
            self.add_node(word, data=node_data)
        return None

    def minus_word_node(self, word: str) -> None:
        if self.has_node(word):
            self.nodes[word]["data"] -= 1
        if self.nodes[word]["data"].get_value() == 0:
            self.remove_node(word)
        return None

    def get_word_node_data(self, word: str) -> None:
        """
        Access the WordNode object for a given word string.
        """
        if self.has_node(word):
            return self.nodes[word]["data"]
        return None

    def _has_edge_with_type(self, u_of_edge: str, v_of_edge: str, edge_type: str):
        """
        Check if an edge with a specific type exists between two nodes.
        """
        # In a MultiDiGraph, has_edge only checks for any edge.
        # We need to check all edges between two nodes for the specific type.
        if not self.has_edge(u_of_edge, v_of_edge):
            return False
        for edge_data in self.get_edge_data(u_of_edge, v_of_edge).values():
            if edge_data.get("type") == edge_type:
                return True
        return False

    def add_semantic_edge(
        self,
        word1: str,
        word2: str,
        weight: float,
        lemma_update: bool = True,
    ) -> None:
        """
        Adds a semantic edge between two words.
        The edge will not expire.
        """
        # Use add_word_node which handles creation
        # Block self loops:
        if word1 == word2:
            return
        if not self.has_node(word1):
            self.add_word_node(word1)
        if not self.has_node(word2):
            self.add_word_node(word2)
        # Add the edge with a weight
        # Edge forms a loop, going both ways
        if weight < self.semantic_threshold:
            return

        if self._has_edge_with_type(word1, word2, "semantic"):
            self.update_semantic_edge(word1, word2, weight)
        else:
            self.add_edge(
                word1, word2, weight=weight, creation=self.time, type="semantic"
            )

        if self._has_edge_with_type(word2, word1, "semantic"):
            self.update_semantic_edge(word2, word1, weight)
        else:
            self.add_edge(
                word2, word1, weight=weight, creation=self.time, type="semantic"
            )

        if lemma_update:
            lemma1 = textUtils.lemmatize_text(word1)[0]
            lemma2 = textUtils.lemmatize_text(word2)[0]
            if lemma1 not in self.embedding_memo:
                self.embedding_memo[lemma1] = textUtils.encode_text(lemma1)
            if lemma2 not in self.embedding_memo:
                self.embedding_memo[lemma2] = textUtils.encode_text(lemma2)
            lemma_weight = textUtils.cosine_similarity(
                self.embedding_memo[lemma1], self.embedding_memo[lemma2]
            )
            self.lemma_graph.add_lemma_edge(lemma1, lemma2, weight=lemma_weight)
        return None

    def update_semantic_edge(self, word1: str, word2: str, weight: float):
        if self._has_edge_with_type(word1, word2, "semantic"):
            # In a multigraph, we need to find the key of the semantic edge
            key_to_update = None
            for key, data in self[word1][word2].items():
                if data.get('type') == 'semantic':
                    key_to_update = key
                    break
            if key_to_update is not None:
                self[word1][word2][key_to_update]["weight"] = weight
        else:
            raise ValueError(f"Semantic edge does not exist between {word1} and {word2}")
        return None

    def add_temporal_edge(self, word1: str, word2: str, weight: float = 1.0):
        """
        Adds a temporal edge between two words.
        """
        if not self.has_node(word1):
            self.add_word_node(word1)
        if not self.has_node(word2):
            self.add_word_node(word2)
        if self._has_edge_with_type(word1, word2, "temporal"):
            # If edge exists, update it only if the new weight is higher
            edge_data = None
            for key, data in self[word1][word2].items():
                if data.get('type') == 'temporal':
                    edge_data = data
                    break
            if edge_data and edge_data["weight"] < weight:
                self.update_temporal_edge(word1, word2, weight=weight)
            return None
        
        self.add_edge(word1, word2, type="temporal", creation=self.time, weight=weight)
        return None

    def update_temporal_edge(self, word1: str, word2: str, weight: float):
        if self._has_edge_with_type(word1, word2, "temporal"):
            key_to_update = None
            for key, data in self[word1][word2].items():
                if data.get('type') == 'temporal':
                    key_to_update = key
                    break
            if key_to_update is not None:
                self[word1][word2][key_to_update]["weight"] = weight
        else:
            raise ValueError(f"Temporal edge does not exist between {word1} and {word2}")
        return None

    def in_out_edges(self, word: str):
        result = {"in": [], "out": []}
        result["in"] = [x for x in self.in_edges(word)]
        result["out"] = [x for x in self.out_edges(word)]
        return result

    def tick(self):
        """
        Ticks the graph forward by one time unit.
        """
        self.time += 1
        return None

    def add_text(
        self,
        text: str,
        yield_frames: bool = False,
        frame_step: int = 1,
        reset_window: bool = False,
    ):
        """
        Adds text to the graph.
        If yield_frames is True, this method is a generator that yields graph states.
        If yield_frames is False, this method runs to completion.
        """
        text_info = textUtils.extract_all_text_info(text)
        words = text_info["words"]
        ending_word_indices = text_info["sentence_ending_words"]
        gen = self._graphUpdate(
            words, ending_word_indices, yield_frames, frame_step, reset_window
        )
        if yield_frames:
            return gen
        else:
            # Consume the generator to run the update to completion
            for _ in gen:
                pass
            return None

    def _graphUpdate(
        self,
        words: list[str],
        ending_word_indices: list[int],
        yield_frames: bool = False,
        frame_step: int = 1,
        reset_window: bool = False,
    ):
        if reset_window:
            self.window = []
        if yield_frames:
            yield self.copy()  # Yield the initial empty graph
        step = 0
        current_index = 0
        for word in words:
            step += 1
            self.add_word_node(word)
            self.window.append(word)
            self.sentence.append(word)

            if len(self.window) > self.text_window_size:
                self.window.pop(0)
            self.tick()
            # Batch-encode any new tokens (current word + existing window tokens)
            to_encode = [
                tok for tok in [word] + self.window if tok not in self.embedding_memo
            ]
            if to_encode:
                self.embedding_memo.update(textUtils.encode_batch(to_encode))
            n = len(self.window) - 1
            for i in range(n):
                prev = self.window[i]
                semantic_weight = textUtils.cosine_similarity(
                    self.embedding_memo[prev], self.embedding_memo[word]
                )
                temporal_weight = sigmoid((n - i) / n)
                self.add_semantic_edge(prev, word, weight=semantic_weight)
                self.add_temporal_edge(prev, word, weight=temporal_weight)
            if ending_word_indices and current_index == ending_word_indices[0]:
                self.semantic_update("sentence")
                ending_word_indices.pop(0)
            if yield_frames and step % frame_step == 0:
                yield self.copy()  # Yield a copy of the graph at each frame step
            current_index += 1

    def semantic_update(self, mode: str):
        """Create semantic edges between **all** tokens currently stored in
        ``self.sentence`` or ``self.paragraph``

        This is a lightweight helper that can be called after you finish
        collecting a sentence or paragraph in a live-streaming scenario.  It
        performs three steps:
        """
        if mode not in ("sentence", "paragraph"):
            raise ValueError("Mode must be 'sentence' or 'paragraph'")

        tokens = self.sentence if mode == "sentence" else self.paragraph
        # Nothing to do for a singleton or empty container.
        if len(tokens) < 2:
            return

        # Batch-encode any unseen tokens to minimise model calls.
        to_encode = [tok for tok in tokens if tok not in self.embedding_memo]
        if to_encode:
            self.embedding_memo.update(textUtils.encode_batch(to_encode))

        # Add semantic edges between each unique unordered pair.
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                w1, w2 = tokens[i], tokens[j]
                weight = textUtils.cosine_similarity(
                    self.embedding_memo[w1], self.embedding_memo[w2]
                )
                self.add_semantic_edge(w1, w2, weight=weight)

        # Optionally clear the lists here after processing – leave to caller.
        self.sentence = [] if mode == "sentence" else self.sentence
        self.paragraph = [] if mode == "paragraph" else self.paragraph

    def jsonify(self):
        data = nx.node_link_data(self, edges="edges")
        json_str = json.dumps(data, cls=NodeEncoder)
        return json_str


def main():
    # text sequence is apple apple banana
    graph = WordGraph()
    graph.add_text("apple apple applesauce")
    graph.add_text("My name is Thomas Cong.")
    print(graph.in_out_edges("apple")["in"][0][1])


if __name__ == "__main__":
    main()
