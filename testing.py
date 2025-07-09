import numpy as np
from scipy.sparse import coo_matrix
from sentence_transformers import SentenceTransformer
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor
model_path = '/Users/tcong/models/all-MiniLM-L6-v2'
model = SentenceTransformer(model_path)

# TODO: Inspect what exactly an SDR is composed of, and investigate what I can do to convert a vector to a SDR for HTM
# TODO: Consider using online EWC as reweighting function for LORA- still monitor anomalies using HTM/other encoding distances

def random_distribution_matrix(d, N, density = 0.1, seed = 42):
    '''
    Generate a random distribution matrix of size d x N with density density
    Args:
    d: dimension of input embedding
    N: output dimension
    density: fraction of non-zero in each column
    seed: random seed for reproducibility
    
    Returns:
    R: random distribution matrix
    '''
    rng = np.random.default_rng(seed)
    # number of non-zero elements
    nnz = int(density * d * N)
    # select random rows that dshould have values assigned there
    rows = rng.integers(d, size = nnz)
    # select random columns that dshould have values assigned there
    cols = rng.integers(N, size = nnz)
    # randomly choose nnz from -1, 1 to assign the row col pairs
    data = rng.choice([-1, 1], size=nnz).astype(np.float32)
    # create a sparse matrix using the previous random sample
    R = coo_matrix((data, (rows, cols)), shape=(d, N), dtype=np.float32)
    # normalize the rows
    R = R / np.sqrt(density * d) 
    return R
def make_sdr(embedding, R, sparsity):
    '''
    Make an SDR from an embedding using a random distribution matrix
    Args:
    embedding: input embedding
    R: random distribution matrix
    sparsity: sparsity of the SDR
    
    Returns:
    SDR: sparse distributed representation
    '''
    # normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    # take the dot product of R and the embedding
    dot_product = embedding @ R
    # number of non-zero elements
    k = int(sparsity * R.shape[1])
    # partition the dot product and return the top k
    idx = np.argpartition(dot_product, -k)[-k:]
    # create an SDR with k non-zero elements
    sdr = np.zeros(R.shape[1], dtype=np.uint8)
    sdr[idx] = 1
    return sdr
def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
if __name__ == "__main__":
    texts = ["Apple", "Computer", "Shit", "Poop"]
    embeddings = model.encode(texts)
    R = random_distribution_matrix(len(embeddings[0]), 4096, density = 0.1, seed = 42)
    # ten percent density SDR
    sdrs = [make_sdr(embedding, R, 0.1) for embedding in embeddings]
    for i, sdr in enumerate(sdrs):
        if i == len(sdrs) - 1:
            break
        print("Texts:", texts[i], texts[i+1])
        print("Embedding Similarity:", cosine_similarity(embeddings[i], embeddings[i+1]))
        print("SDR Similarity:", cosine_similarity(sdr, sdrs[i+1]))
    

    
