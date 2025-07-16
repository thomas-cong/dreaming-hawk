import networkx as nx
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
    def __init__(self, expiration_time=30, semantic_threshold = 0.5):
        super().__init__()
        self.expiration_time = expiration_time
        self.semantic_threshold = semantic_threshold
        self.time = 0
        self.expiration_object_dict = {}
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
        # Add the edge with a weight attribute
        if weight >= self.semantic_threshold:
            self.add_edge(word1, word2, weight=weight)
        return None
        

    def add_temporal_edge(self, word1, word2, expiration=None):
        '''
        Adds a temporal edge between two words.
        The edge will expire after the specified expiration time.
        If expiration is None, the edge will expire after the expiration time of the graph.
        '''
        if expiration is None:
            expiration = self.expiration_time
        if not self.has_node(word1):
            self.add_word_node(word1)
        if not self.has_node(word2):
            self.add_word_node(word2)
        self.add_edge(word1, word2, type="temporal", expiration=self.time + expiration)
        if self.time + expiration not in self.expiration_object_dict:
            self.expiration_object_dict[self.time + expiration] = []
        self.expiration_object_dict[self.time + expiration].append((word1, word2))
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
                        if data.get('type') == 'temporal' and data.get('expiration') == self.time:
                            key_to_remove = key
                            break # Remove one edge per tick
                if key_to_remove is not None:
                    self.remove_edge(u, v, key=key_to_remove)
            del self.expiration_object_dict[self.time]

def main():
    # text sequence is apple apple banana
    wg = WordGraph(expiration_time=5)
    wg.add_word_node("apple")
    wg.add_word_node("apple")
    wg.add_word_node("banana")
    wg.add_semantic_edge("apple", "banana", weight=0.8)
    wg.add_temporal_edge("apple", "banana")
    print(f"Edges before tick: {list(wg.edges())}")
    wg.tick()
    print(f"Edges after 1st tick: {list(wg.edges())}")
    for _ in range(5):
        wg.tick()
    print(f"Edges after 6 ticks: {list(wg.edges())}")

if __name__ == '__main__':
    main()
