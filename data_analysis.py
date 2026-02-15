import pandas as pd

# Data organized manually from the LaTeX tables
data = [
    # Text-only: IEMOCAP
    ["Text-only", "IEMOCAP", "MSP-IMPROV", 41.48, 0.3, 41.27, 0.5],
    ["Text-only", "IEMOCAP", "MSP-PODCAST", 40.09, 0.2, 39.68, 0.4],
    ["Text-only", "IEMOCAP", "SAMSEMO-EN", 47.27, 0.7, 46.36, 0.8],
    ["Text-only", "IEMOCAP", "CMU-MOSEI", 41.26, 0.8, 42.59, 1.1],

    # Text-only: MSP-IMPROV
    ["Text-only", "MSP-IMPROV", "IEMOCAP", 45.91, 0.55, 45.32, 0.67],
    ["Text-only", "MSP-IMPROV", "MSP-PODCAST", 37.24, 0.72, 37.42, 0.76],
    ["Text-only", "MSP-IMPROV", "SAMSEMO-EN", 40.85, 1.33, 41.73, 1.32],
    ["Text-only", "MSP-IMPROV", "CMU-MOSEI", 40.41, 1.17, 40.37, 1.73],

    # Text-only: MSP-PODCAST
    ["Text-only", "MSP-PODCAST", "IEMOCAP", 45.30, 0.32, 45.73, 0.35],
    ["Text-only", "MSP-PODCAST", "MSP-IMPROV", 39.05, 0.09, 39.37, 0.12],
    ["Text-only", "MSP-PODCAST", "CMU-MOSEI", 46.26, 0.84, 48.55, 0.70],
    ["Text-only", "MSP-PODCAST", "SAMSEMO-EN", 49.09, 0.32, 51.12, 0.16],

    # Audio-only: IEMOCAP
    ["Audio-only", "IEMOCAP", "MSP-IMPROV", 45.02, 1.42, 48.89, 0.62],
    ["Audio-only", "IEMOCAP", "MSP-PODCAST", 37.67, 0.70, 38.97, 0.51],
    ["Audio-only", "IEMOCAP", "SAMSEMO-EN", 45.19, 1.44, 46.96, 0.28],
    ["Audio-only", "IEMOCAP", "CMU-MOSEI", 40.41, 1.17, 40.37, 1.73],

    # Audio-only: MSP-IMPROV
    ["Audio-only", "MSP-IMPROV", "IEMOCAP", 54.92, 1.48, 58.17, 0.56],
    ["Audio-only", "MSP-IMPROV", "MSP-PODCAST", 36.87, 0.89, 37.49, 0.44],
    ["Audio-only", "MSP-IMPROV", "SAMSEMO-EN", 41.87, 1.32, 41.49, 0.63],
    ["Audio-only", "MSP-IMPROV", "CMU-MOSEI", 36.75, 0.47, 35.98, 0.52],

    # Audio-only: MSP-PODCAST
    ["Audio-only", "MSP-PODCAST", "IEMOCAP", 54.25, 0.80, 55.55, 0.66],
    ["Audio-only", "MSP-PODCAST", "MSP-IMPROV", 48.97, 0.62, 50.57, 0.64],
    ["Audio-only", "MSP-PODCAST", "SAMSEMO-EN", 53.11, 0.42, 54.57, 0.15],
    ["Audio-only", "MSP-PODCAST", "CMU-MOSEI", 44.12, 0.31, 45.27, 0.56],

    # Fusion: IEMOCAP
    ["Fusion", "IEMOCAP", "MSP-IMPROV", 48.05, 0.48, 48.56, 0.27],
    ["Fusion", "IEMOCAP", "MSP-PODCAST", 41.84, 0.30, 42.15, 0.15],
    ["Fusion", "IEMOCAP", "SAMSEMO-EN", 45.92, 0.35, 46.69, 0.81],
    ["Fusion", "IEMOCAP", "CMU-MOSEI", 42.47, 0.69, 43.50, 1.35],

    # Fusion: MSP-IMPROV
    ["Fusion", "MSP-IMPROV", "IEMOCAP", 58.74, 0.55, 59.16, 0.58],
    ["Fusion", "MSP-IMPROV", "MSP-PODCAST", 41.36, 0.52, 42.21, 0.32],
    ["Fusion", "MSP-IMPROV", "SAMSEMO-EN", 43.19, 0.37, 42.89, 0.24],
    ["Fusion", "MSP-IMPROV", "CMU-MOSEI", 44.75, 1.27, 45.58, 1.62],

    # Fusion: MSP-PODCAST
    ["Fusion", "MSP-PODCAST", "IEMOCAP", 53.82, 0.99, 57.88, 0.01],
    ["Fusion", "MSP-PODCAST", "MSP-IMPROV", 48.84, 0.43, 52.17, 0.21],
    ["Fusion", "MSP-PODCAST", "SAMSEMO-EN", 53.18, 0.10, 58.98, 0.27],
    ["Fusion", "MSP-PODCAST", "CMU-MOSEI", 44.64, 0.41, 54.28, 0.05],
]

df = pd.DataFrame(data, columns=[
    "Modality", "Train Dataset", "Test Dataset",
    "Baseline Mean", "Baseline Std", "Full Mean", "Full Std"
])

csv_path = "/mnt/data/cross_corpus_results.csv"
df.to_csv(csv_path, index=False)

csv_path
