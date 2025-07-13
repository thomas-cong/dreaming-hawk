from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor
from htm.bindings.sdr import SDR
from textParsing import parse_text
from embeddingSDRs import make_binary_vectors, random_distribution_matrix
import os
from dotenv import load_dotenv
load_dotenv(".env")

HTM_DIM = int(os.getenv("HTM_DIM"))
print("HTM_DIM:", HTM_DIM)
# HTM setup (parameters borrowed from the Numenta Hotgym example)

# Hyper-parameters for the Spatial Pooler, Temporal Memory, etc.
default_parameters = {
    'sp': {
        'boostStrength': 3.0,
        'columnCount': HTM_DIM,
        'localAreaDensity': 0.04395604395604396,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006,
    },
    'tm': {
        'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1,
    },
    'predictor': {'sdrc_alpha': 0.1},
    'anomaly': {'period': 1000},
}

# Use the SDR length (HTM_DIM) produced by our embedding→SDR pipeline.
encodingWidth = HTM_DIM
spParams = default_parameters['sp']
tmParams = default_parameters['tm']

# Spatial Pooler
print("Starting spatial pooler")
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
print("Finished spatial pooler")
# Temporal Memory
print("Starting temporal memory")
tm = TemporalMemory(
    columnDimensions=(spParams['columnCount'],),
    cellsPerColumn=tmParams['cellsPerColumn'],
    activationThreshold=tmParams['activationThreshold'],
    initialPermanence=tmParams['initialPerm'],
    connectedPermanence=spParams['synPermConnected'],
    minThreshold=tmParams['minThreshold'],
    maxNewSynapseCount=tmParams['newSynapseCount'],
    permanenceIncrement=tmParams['permanenceInc'],
    permanenceDecrement=tmParams['permanenceDec'],
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell=tmParams['maxSegmentsPerCell'],
    maxSynapsesPerSegment=tmParams['maxSynapsesPerSegment'],
)
print("Finished temporal memory")
# Anomaly Likelihood & Predictor
print("Starting anomaly likelihood")
anomaly_history = AnomalyLikelihood(default_parameters['anomaly']['period'])
predictor = Predictor(steps=[1, 5], alpha=default_parameters['predictor']['sdrc_alpha'])
predictor_resolution = 10
print("Finished anomaly likelihood")
if __name__ == "__main__":
    words = parse_text("/Users/tcong/dreaming-hawk/TrainingTexts/ChalmersPaper.txt", mode = 'words')
    R = random_distribution_matrix(len(words[0]), HTM_DIM, density = 0.1, seed = 42)
    binary_vectors = make_binary_vectors(words, density = 0.1)
    anomaly = []
    anomaly_probs = []
    for t, vector in enumerate(binary_vectors):
        # Create a empty SDR of size HTM_DIM
        sdr = SDR([HTM_DIM])
        # Set the SDR to the binary vector
        sdr.dense = vector
        # Compute the active columns
        activeColumns = SDR(sp.getColumnDimensions())
        sp.compute(sdr, True, activeColumns)
        # Compute the anomaly
        tm.compute(activeColumns, True)
        instant = tm.anomaly
        likelihood = anomaly_history.compute(instant)
        anomaly.append(instant)
        anomaly_probs.append(likelihood)
    
    