# DDoS Big Data Analytics and Recommendation System

**Student:** Benjamine  
**Programme:** MSc Artificial Intelligence, Batch 18  
**University:** University of Moratuwa  
**Module:** Big Data Analytics Mini Project

---

## Project Overview

This project implements a complete big data analytics and recommendation system
on the BCCC-cPacket-Cloud-DDoS-2024 dataset, using Apache Spark (PySpark) for
all distributed data processing.

**Part A** — Big Data Analytics on 540,494 DDoS network flow records across
319 features. Answers five analytical questions using SparkSQL, DataFrame API,
and window functions.

**Part B** — DDoS attack type recommendation system. Given an observed attack
on a network service, recommends which other attack types to prepare defences
for. Implements four recommendation layers: collaborative filtering (Item-KNN),
ALS matrix factorization (pyspark.ml), content-based similarity, and
LLM reflection motivated by CRAG (Zhu et al., WWW 2025).

---

## Dataset

**Name:** BCCC-cPacket-Cloud-DDoS-2024  
**Source:** York University / cPacket Networks  
**Access:** Kaggle — dhoogla/bccc-cpacket-cloud-ddos-2024  
**License:** CC-BY-SA-4.0  
**Size:** 540,494 flows × 319 features  
**Format:** Parquet (29.5 MB compressed)

The dataset is not committed to this repository. It is downloaded automatically
by the notebooks via the Kaggle API on first run.

---

## Environment Setup

### Prerequisites
- Python 3.9 or higher
- Java 17 (required by PySpark)
- Git

### Installation

Clone the repository and set up the virtual environment:

    git clone https://github.com/your-username/ddos-bigdata-project
    cd ddos-bigdata-project
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    export JAVA_HOME=$(brew --prefix openjdk@17)

### Kaggle API Setup

1. Go to kaggle.com → Settings → API → Create New Token
2. Set as environment variable: export KAGGLE_TOKEN=your_token_here

### OpenAI API Setup

    export OPENAI_API_KEY=your_key_here

---

## Running the Notebooks

### Option 1 — Jupyter in browser

    jupyter notebook

Open notebooks/part_a_analytics.ipynb then notebooks/part_b_recommendation.ipynb.
Run all cells top to bottom in each notebook.

### Option 2 — VS Code

Open VS Code in the project folder. Select the venv kernel.
Open and run each notebook using the VS Code Jupyter extension.

### Option 3 — Google Colab (original execution environment)

Upload notebooks to Colab. Add KAGGLE_TOKEN and OPENAI_API_KEY to Colab Secrets.
Run all cells top to bottom.

### Option 4 — Streamlit UI

    streamlit run app/recommendation_ui.py

Opens at http://localhost:8501

---

## Project Structure

    ddos-bigdata-project/
    ├── notebooks/
    │   ├── part_a_analytics.ipynb
    │   └── part_b_recommendation.ipynb
    ├── app/
    │   └── recommendation_ui.py
    ├── outputs/
    ├── data/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Techniques Used

### Part A

SparkSQL and DataFrame API on 540,494 network flows:
- Parquet loading with schema preservation
- Null audit across 319 columns
- Window functions: RANK() OVER, SUM() OVER()
- GroupBy aggregation with programmatic expression building
- CASE WHEN port mapping
- TCP flag heatmap normalisation
- Feature discrimination ratio analysis

### Part B

Recommendation system layers:
- Item-KNN collaborative filtering (cosine similarity on co-occurrence matrix)
- ALS matrix factorization via pyspark.ml.recommendation
- Content-based similarity (cosine similarity on Part A feature centroids)
- Hybrid weighted combination (CF + content)
- LLM reflection motivated by CRAG (Zhu et al., WWW 2025)
- Evaluation: Precision@K, Recall@K, RMSE

---

## References

Zhu, Y. et al. (2025). Collaborative Retrieval for Large Language Model-based
Conversational Recommender Systems. WWW 2025.

York University / cPacket Networks (2024). BCCC-cPacket-Cloud-DDoS-2024 Dataset.
Kaggle. https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024