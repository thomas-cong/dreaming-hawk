import htm
import htm.bindings.encoders as encoders
from sentence_transformers import SentenceTransformer

model_path = "/Users/tcong/models/MiniLM-L6-V2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
model = SentenceTransformer(model_path)
text = "Brainrot"
embeddings = model.encode([text])

# TODO: Inspect what exactly an SDR is composed of, and investigate what I can do to convert a vector to a SDR for HTM
