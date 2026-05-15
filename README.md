# DDoS Big Data Analytics and Recommendation System

**Student:** Benjamine. S (258762A)
**Batch:** 18
**Programme:** MSc Artificial Intelligence
**University:** University of Moratuwa
**Module:** IT5612 — Big Data Analytics Mini Project

---

## About This Project

I completed both Part A and Part B of the assignment using the
BCCC-cPacket-Cloud-DDoS-2024 dataset — a real DDoS network traffic
dataset published by York University and cPacket Networks in 2024.

**Part A** answers five analytical questions on 540,494 network flow
records using Apache Spark.

**Part B** builds a recommendation system that tells a SOC analyst
which DDoS attack types to prepare defences for, given an observed
attack on a network service.

---

## Dataset

**Name:** BCCC-cPacket-Cloud-DDoS-2024
**Source:** York University / cPacket Networks
**Link:** https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024
**License:** CC-BY-SA-4.0
**Size:** 540,494 flows, 319 features per flow

The dataset is not in this repository. The Part A notebook downloads
it automatically using the Kaggle API when you run it for the first time.

---

## What Is in This Repository


---

## How to Run

### Requirements

- Python 3.9 or higher
- Java 17
- Git

### Setup

```bash
git clone https://github.com/ITToday/ddos-bigdata-project
cd ddos-bigdata-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export JAVA_HOME=$(brew --prefix openjdk@17)
```

### API Keys

You need two API keys before running the notebooks.

**Kaggle** — to download the dataset:
Go to kaggle.com → Settings → API → Create New Token

**OpenAI** — for the LLM layer in Part B:
Go to platform.openai.com → API Keys

```bash
export KAGGLE_TOKEN=your_token_here
export OPENAI_API_KEY=your_key_here
```

### Running Part A

```bash
jupyter notebook
```

Open `notebooks/part_a_analytics.ipynb` and run all cells from top
to bottom. It will download the dataset, run the analysis, and save
all charts and results to the `outputs/` folder.

### Running Part B

Open `notebooks/part_b_recommendation.ipynb` and run all cells from
top to bottom. It loads the Part A outputs and builds the recommendation
system.

### Running the Streamlit UI

```bash
streamlit run app/recommendation_ui.py
```

Opens at http://localhost:8501. No Spark needed here — it reads the
precomputed results from the `outputs/` folder.

### Running on Google Colab

Upload both notebooks to Colab. Add KAGGLE_TOKEN and OPENAI_API_KEY
to Colab Secrets using the key icon in the left sidebar. Run all cells
from top to bottom.

---

## References

Zhu, Y. et al. (2025). Collaborative Retrieval for Large Language
Model-based Conversational Recommender Systems. WWW 2025, Sydney.

York University / cPacket Networks (2024). BCCC-cPacket-Cloud-DDoS-2024.
https://www.kaggle.com/datasets/dhoogla/bccc-cpacket-cloud-ddos-2024