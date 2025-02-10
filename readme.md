# Music Recommender System

## Description
The Music Recommender System is a scalable project that builds a recommendation engine for music tracks. The project began as a proof-of-concept (POC) that used a CSV file containing a small dataset (initially 40 songs, later updated to 200 songs) with basic similarity calculations. This version was later evolved into a final implementation that leverages a SQLite database and FAISS (Facebook AI Similarity Search) to handle large datasets (up to 1 million songs). The system employs batch processing, memory mapping, and efficient similarity search techniques to generate personalized recommendations based on user interactions and song features.

## Features
- **Proof-of-Concept (POC):**
  - Uses a CSV file with basic similarity calculations.
- **Final Implementation:**
  - Uses a SQLite database and FAISS for large-scale similarity search.
  - Efficient data loading through batch processing and memory mapping.
- **Performance Evaluation:**
  - A `performance.py` script is provided to compare key metrics (data load time, memory usage, and recommendation generation time) between the POC and final implementations.
- **Recommendation Engine:**
  - Generates recommendations based on a hybrid approach combining content similarity and popularity metrics.

## Prerequisites
- Python 3.7 or later
- Required Python packages (listed in `requirements.txt`):
  - numpy
  - faiss-cpu
  - psutil
  - matplotlib
  - pandas
  - networkx
  - scikit-learn

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ImAsrith/MSCS532_Project.git
   cd MSCS532_Project
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Program

### Proof-of-Concept (POC)
The POC version uses a CSV file (`songs.csv`) containing approximately 200 songs. To run the POC implementation:
```bash
python music_recommender.py
```

### Final Implementation
The final version uses a SQLite database (`million_songs.db`) and FAISS for similarity search. To run the final implementation:
```bash
python music_recommender_v1.py
```

### Performance Evaluation
To compare the performance of the POC and final implementations, run:
```bash
python performance.py
```
The performance script measures data load time, memory usage, and recommendation generation time. It also extrapolates the POC metrics to simulate a dataset of 1 million songs for comparison.

## Project Structure
- **music_recommender.py:** Proof-of-Concept implementation using a CSV file.
- **music_recommender_v1.py:** Final implementation using a SQLite database and FAISS.
- **performance.py:** Script for performance evaluation and visualization.
- **requirements.txt:** List of required Python packages.
- **readme.md:** Project description and installation instructions.

## Performance Analysis (Overview)
- **Data Load Time:**
  - The POC version loads 200 songs in approximately 0.010 seconds (extrapolated to 51.81 seconds for 1 million songs).
  - The final implementation loads 1 million songs in about 9.847 seconds.
- **Memory Usage:**
  - The extrapolated memory usage for the POC version is estimated at 4375.00 MB.
  - The final implementation uses approximately 701.95 MB.
- **Recommendation Generation Time:**
  - The POC version's recommendation time is nearly 0 ms on a small dataset (extrapolated to 8.73 ms for 1 million songs).
  - The final implementation shows a recommendation time of about 2131.31 ms.
  
_**Note:** Extrapolated values assume linear scaling and should be interpreted with caution._

_Leave space here for three graphs: one for data load time, one for memory usage, and one for recommendation generation time._

## Future Work
- Optimize the recommendation generation process.
- Incorporate additional features for improved similarity assessment.
- Explore advanced collaborative filtering techniques.
