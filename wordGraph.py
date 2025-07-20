import networkx as nx
from tqdm import tqdm
import textUtils

class WordNodeData:
    def __init__(self, word, value):
        self.word = word
        self.value = value

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def __str__(self):
        return "WordNode(" + self.word + ", " + str(self.value) + ")"

    def __iadd__(self, other):
        self.value += other
        return self

    def __hash__(self):
        return hash(self.word)


class WordGraph(nx.MultiDiGraph):
    def __init__(self, text_window_size=30, semantic_threshold = 0.5):
        super().__init__()
        self.text_window_size = text_window_size
        self.semantic_threshold = semantic_threshold
        self.time = 0
        self.expiration_object_dict = {}
        self.embedding_memo = {}
        self.window = []

    def get_window(self):
        return self.window

    def get_time(self):
        return self.time

    def add_word_node(self, word):
        '''
        Adds a word to the graph or increments its value if it already exists.
        '''
        if self.has_node(word):
            # Access the data via the string `word`
            self.nodes[word]['data'] += 1
        else:
            # Use the string `word` as the node and store WordNode in an attribute
            node_data = WordNodeData(word, 1)
            self.add_node(word, data=node_data)
        return None

    def get_word_node_data(self, word):
        '''
        Access the WordNode object for a given word string.
        '''
        if self.has_node(word):
            return self.nodes[word]['data']
        return None

    def add_semantic_edge(self, word1, word2, weight):
        '''
        Adds a semantic edge between two words.
        The edge will not expire.
        '''
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
        if weight >= self.semantic_threshold:
            self.add_edge(word1, word2, weight=weight)
            self.add_edge(word2, word1, weight=weight)
        return None

    def add_temporal_edge(self, word1, word2, duration=None):
        '''
        Adds a temporal edge between two words.
        The edge will expire after the specified duration.
        If duration is None, the edge will expire after the text window size of the graph.
        '''
        if duration is None:
            duration = self.text_window_size
        if not self.has_node(word1):
            self.add_word_node(word1)
        if not self.has_node(word2):
            self.add_word_node(word2)
        self.add_edge(word1, word2, type="temporal", expiration_time=self.time + duration)
        if self.time + duration not in self.expiration_object_dict:
            self.expiration_object_dict[self.time + duration] = []
        self.expiration_object_dict[self.time + duration].append((word1, word2))
        return None

    def tick(self):
        '''
        Ticks the graph forward by one time unit.
        Removes any expired edges.
        '''
        self.time += 1
        if self.time in self.expiration_object_dict:
            for u, v in self.expiration_object_dict[self.time]:
                # In a MultiDiGraph, we need to find the specific edge to remove.
                # We do this by finding an edge with the 'temporal' type.
                key_to_remove = None
                if self.has_edge(u, v):
                    for key, data in self.get_edge_data(u, v).items():
                        if data.get('type') == 'temporal' and data.get('expiration_time') == self.time:
                            key_to_remove = key
                            break # Remove one edge per tick
                if key_to_remove is not None:
                    self.remove_edge(u, v, key=key_to_remove)
            del self.expiration_object_dict[self.time]

    def add_text(self, text, yield_frames=False, frame_step=1, reset_window=False):
        '''
        Adds text to the graph.
        If yield_frames is True, this method is a generator that yields graph states.
        If yield_frames is False, this method runs to completion.
        '''
        if isinstance(text, str):
            words = textUtils.split_text(text, mode='words')
        else:
            words = text
        gen = self._graphUpdate(words, yield_frames, frame_step, reset_window)
        if yield_frames:
            return gen
        else:
            # Consume the generator to run the update to completion
            for _ in gen:
                pass
            return None
    def _graphUpdate(self, words, yield_frames=False, frame_step=1, reset_window=False):
        if reset_window:
            self.window = []
        step = 0
        if yield_frames:
            yield self.copy() # Yield the initial empty graph
        for word in tqdm(words):
            step += 1
            self.add_word_node(word)
            self.tick()
            if word not in self.embedding_memo:
                self.embedding_memo[word] = textUtils.encode_text(word)
            for prev in self.window:
                if prev not in self.embedding_memo:
                    self.embedding_memo[prev] = textUtils.encode_text(prev)
                weight = textUtils.cosine_similarity(self.embedding_memo[prev], self.embedding_memo[word])
                self.add_semantic_edge(prev, word, weight=weight)
                self.add_temporal_edge(prev, word)
            self.window.append(word)
            if len(self.window) > self.text_window_size:
                self.window.pop(0)
            if yield_frames and step % frame_step == 0:
                yield self.copy() # Yield a copy of the graph at each frame step
    def enrich_semantic_connections(self, text, yield_frames=False, frame_step=1, sentence_weighting = 0.9, paragraph_weighting = 0.6):
        gen = self._enrich_semantic_connections(text, sentence_weighting, paragraph_weighting)
        if yield_frames:
            return gen
        else:
            # Consume the generator to run the update to completion
            for _ in gen:
                pass
            return None
    def _enrich_semantic_connections(self, text, sentence_weighting = 0.9, paragraph_weighting = 0.6):
        '''
        Enriches the graph with semantic connections between words.
        '''
        if not text:
            raise ValueError("Text must not be empty")
        words_not_found = 0
        # Parse sentences, and semantically link words within sentences
        sentences = textUtils.split_text(text, mode='sentences')
        for sentence in tqdm(sentences):
            word_list = []
            sentence_words = textUtils.split_text(sentence, mode='words')
            for word in tqdm(sentence_words):
                if word not in self.nodes:
                    self.add_word_node(word)
                if word not in self.embedding_memo:
                    words_not_found += 1
                    self.embedding_memo[word] = textUtils.encode_text(word)
                for i in range(len(word_list)):
                    weight = textUtils.cosine_similarity(self.embedding_memo[word_list[i]], self.embedding_memo[word])
                    self.add_semantic_edge(word_list[i], word, weight=weight * sentence_weighting)
                    yield self.copy()
                word_list.append(word)
        print("Words not found: " + str(words_not_found))
    
                    

def main():
    # text sequence is apple apple banana
    wg = WordGraph(text_window_size=5)

if __name__ == '__main__':
    main()
