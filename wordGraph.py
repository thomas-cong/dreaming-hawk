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
class WordGraph(nx.DiGraph):
    def __init__(self, expiration_time=30):
        super().__init__()
        self.expiration_time = expiration_time
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
        self.add_word_node(word1)
        self.add_word_node(word2)
        # Add the edge with a weight attribute
        self.add_edge(word1, word2, weight=weight)

    def add_temporal_edge(self, word1, word2, expiration=None):
        '''
        Adds a temporal edge between two words.
        The edge will expire after the specified expiration time.
        If expiration is None, the edge will expire after the expiration time of the graph.
        '''
        if expiration is None:
            expiration = self.expiration_time
        self.add_edge(word1, word2, type="temporal", expiration=self.time + expiration)
        if self.time + expiration not in self.expiration_object_dict:
            self.expiration_object_dict[self.time + expiration] = []
        self.expiration_object_dict[self.time + expiration].append(word2)

    def tick(self):
        '''
        Ticks the graph forward by one time unit.
        Removes any expired nodes and edges.
        '''
        self.time += 1
        if self.time in self.expiration_object_dict:
            for object in self.expiration_object_dict[self.time]:
                self.remove_node(object)

if __name__ == '__main__':
    # text sequence is apple apple banana
    wg = WordGraph(expiration_time=5)
    wg.add_word_node("apple")
    wg.add_word_node("apple")
    wg.add_word_node("banana")
    wg.add_semantic_edge("apple", "banana", weight=0.8)
    wg.add_temporal_edge("apple", "banana")
    
