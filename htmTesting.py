from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.sdr import SDR
from textParsing import parse_text, sliding_window
from embeddingSDRs import make_binary_vectors, random_distribution_matrix
import os
import pickle
from dotenv import load_dotenv
load_dotenv(".env")

HTM_DIM = int(os.getenv("HTM_DIM"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))
print("HTM_DIM:", HTM_DIM)
print("EMBEDDING_DIM:", EMBEDDING_DIM)
# HTM setup (parameters borrowed from the Numenta Hotgym example)
# TODO: Adjust parameters and see if performance improves
# Hyper-parameters for the Spatial Pooler, Temporal Memory, etc.
default_parameters = {
    'sp': {
        'boostStrength': 1.5,
        'columnCount': HTM_DIM,
        'localAreaDensity': 0.02,
        'potentialPct': 0.5,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13,
        'synPermInactiveDec': 0.006,
        'seed': 42},
    'tm': {
        'activationThreshold': 15,
        'cellsPerColumn': 8,
        'initialPerm': 0.21,
        'connectedPerm': 0.3,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 15,
        'maxNewSynapseCount': 40,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1,
        'predictedSegmentDecrement': 0.01,
        'seed': 42
    },
    'anomaly': {'period': 300},
}

# Use the SDR length (HTM_DIM) produced by our embedding→SDR pipeline.
encodingWidth = HTM_DIM
spParams = default_parameters['sp']
tmParams = default_parameters['tm']

# Spatial Pooler
print("Spinning up spatial pooler")
sp = SpatialPooler(
    inputDimensions=(encodingWidth,),
    columnDimensions=(spParams['columnCount'],),
    potentialPct=spParams['potentialPct'],
    potentialRadius=encodingWidth,
    globalInhibition=True,
    localAreaDensity=spParams['localAreaDensity'],
    synPermInactiveDec=spParams['synPermInactiveDec'],
    synPermActiveInc=spParams['synPermActiveInc'],
    synPermConnected=spParams['synPermConnected'],
    boostStrength=spParams['boostStrength'],
    wrapAround=True,
)
print("Instantiated spatial pooler")
# Temporal Memory
print("Spinning up temporal memory")
tm = TemporalMemory(
    columnDimensions=(spParams['columnCount'],),
    cellsPerColumn=tmParams['cellsPerColumn'],
    activationThreshold=tmParams['activationThreshold'],
    initialPermanence=tmParams['initialPerm'],
    connectedPermanence=tmParams['connectedPerm'],
    minThreshold=tmParams['minThreshold'],
    maxNewSynapseCount=tmParams['maxNewSynapseCount'],
    permanenceIncrement=tmParams['permanenceInc'],
    permanenceDecrement=tmParams['permanenceDec'],
    predictedSegmentDecrement=tmParams['predictedSegmentDecrement'],
    maxSegmentsPerCell=tmParams['maxSegmentsPerCell'],
    maxSynapsesPerSegment=tmParams['maxSynapsesPerSegment'],
)
print("Instantiated temporal memory")
# Anomaly Likelihood & Predictor
print("Spinning up anomaly likelihood")
anomaly_history = AnomalyLikelihood(default_parameters['anomaly']['period'])
print("Instantiated anomaly likelihood")
if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import tqdm
    import numpy as np
    import pickle

    # Define model file paths
    MODEL_DIR = "models"
    SP_PATH = os.path.join(MODEL_DIR, "sp_model.dat")
    TM_PATH = os.path.join(MODEL_DIR, "tm_model.dat")
    ANOMALY_PATH = os.path.join(MODEL_DIR, "anomaly_model.pkl")

    # Create directory for models if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Load models if they exist, otherwise train them
    if all(os.path.exists(p) for p in [SP_PATH, TM_PATH, ANOMALY_PATH]):
        print("Loading models from disk...")
        sp.loadFromFile(SP_PATH)
        tm.loadFromFile(TM_PATH)
        with open(ANOMALY_PATH, "rb") as f:
            anomaly_history = pickle.load(f)
        print("Models loaded.")
    else:
        print("Could not find saved models. Training a new model...")
        canon = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/FullSherlockHolmes.txt", mode='words')
        print("Canonical text length:", len(canon))
        # Build sliding windows with stride to reduce total examples
        WINDOW_SIZE = 4
        STRIDE = 3
        R = random_distribution_matrix(EMBEDDING_DIM, HTM_DIM, density=0.05, seed=42)
        canon_windows = list(sliding_window(canon, window_size=WINDOW_SIZE, stride=STRIDE))
        print("Canonical windows length:", len(canon_windows))
        training_set = make_binary_vectors(canon_windows, R=R, density=0.05)
        print("Training set length:", len(training_set))
        # --- Training Phase ---
        for epoch in range(5):
            print(f"Epoch {epoch+1}/5...")
            for vector in tqdm.tqdm(training_set, total=len(training_set)):
                sdr = SDR([HTM_DIM])
                sdr.dense = vector
                activeColumns = SDR(sp.getColumnDimensions())
                sp.compute(sdr, True, activeColumns)
                tm.compute(activeColumns, True)

        # --- Calibration Phase ---
        print("Calibrating anomaly likelihood...")
        for vector in tqdm.tqdm(training_set, total=len(training_set)):
            sdr = SDR([HTM_DIM])
            sdr.dense = vector
            activeColumns = SDR(sp.getColumnDimensions())
            sp.compute(sdr, False, activeColumns)
            tm.compute(activeColumns, False)
            anomaly_history.compute(tm.anomaly)

        # --- Save Models ---
        print("Saving models to disk...")
        sp.saveToFile(SP_PATH)
        tm.saveToFile(TM_PATH)
        with open(ANOMALY_PATH, "wb") as f:
            pickle.dump(anomaly_history, f)
        print("Models saved.")

    WINDOW_SIZE = 10
    STRIDE = 6
    # --- Testing Phase ---
    anomalous = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/AnomalyTest.txt", mode='words')
    anomalous_windows = list(sliding_window(anomalous, window_size=WINDOW_SIZE, stride=STRIDE))
    # We need the random matrix R for the testing set as well
    R = random_distribution_matrix(EMBEDDING_DIM, HTM_DIM, density=0.05, seed=42)
    testing_set = make_binary_vectors(anomalous_windows, R=R, density=0.05)
    
    anomaly_probs = []
    anomaly = []
    for t, vector in tqdm.tqdm(enumerate(testing_set), total = len(testing_set)):
        # Create a empty SDR of size HTM_DIM
        sdr = SDR([HTM_DIM])
        # Set the SDR to the binary vector
        sdr.dense = vector
        # Compute the active columns
        activeColumns = SDR(sp.getColumnDimensions())
        sp.compute(sdr, False, activeColumns)
        # Compute the anomaly
        tm.compute(activeColumns, False)
        instant = tm.anomaly
        likelihood = anomaly_history.compute(instant)
        anomaly.append(instant)
        anomaly_probs.append(likelihood)
    
    sns.lineplot(x = range(len(anomaly)), y = anomaly, label="Instantaneous Anomaly Score")
    sns.lineplot(x = range(len(anomaly_probs)), y = anomaly_probs, label="Anomaly Likelihood")
    plt.legend()
    plt.show()
    
    