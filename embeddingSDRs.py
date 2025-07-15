import numpy as np
import os
from scipy.sparse import coo_matrix
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv(".env")
HTM_DIM = int(os.environ["HTM_DIM"])
model_path = '/Users/tcong/models/all-MiniLM-L6-v2'
model = SentenceTransformer(model_path)
print("HTM_DIM:", HTM_DIM)
def random_distribution_matrix(d: int, N: int, density: float = 0.1, seed: int = 42) -> coo_matrix:
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
def make_sdr(embedding: np.ndarray, R: coo_matrix, sparsity: float) -> list[int]:
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
    return tuple(sdr)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
def make_binary_vectors(text: list[str], density: float = 0.1, R: coo_matrix = None) -> list[int]:
    '''
    Make an SDR from a text using a random distribution matrix
    Args:
    text: input text
    
    Returns:
    SDR: sparse distributed representation
    '''
    embeddings = model.encode(text)
    if R is None:
        R = random_distribution_matrix(len(embeddings[0]), HTM_DIM, density = density, seed = 42)
    # ten percent density SDR
    sdrs = []
    print("Making SDRS from embeddings")
    for embedding in tqdm(embeddings):
        sdr = make_sdr(embedding, R, density)
        sdrs.append(sdr)
    return sdrs
if __name__ == "__main__":
    texts = ["Apple", "Computer", "Shit", "Poop"]
    embeddings = model.encode(texts)
    R = random_distribution_matrix(len(embeddings[0]), 1024, density = 0.1, seed = 42)
    # ten percent density SDR
    binary_vectors = make_binary_vectors(texts, density = 0.1)
    for i, vector in enumerate(binary_vectors):
        if i == len(binary_vectors) - 1:
            break
        print("Texts:", texts[i], texts[i+1])
        print("Embedding Similarity:", cosine_similarity(embeddings[i], embeddings[i+1]))
        print("Binary Vector (SDR) Similarity:", cosine_similarity(vector, binary_vectors[i+1]))
    

    
