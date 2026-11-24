# Public Sentiment Analysis on Platform X Toward Indonesian National Police (IndoBERT)

An end-to-end Natural Language Processing (NLP) project analyzing public perception of the Indonesian National Police (Polri) on Platform X (formerly Twitter) between January 2025 and October 2025. Built using a fine-tuned **IndoBERT** deep learning model and deployed as an interactive **Streamlit** dashboard.

## Project Overview & Key Metrics
* **Data Collection & Cleaning:** Scraped **7,444 raw tweets** via Google Colab and performed deduplication/noise removal to retain **6,456 unique, valid records**.
* **Ground Truth Labeling:** Manually labeled **1,000 balanced samples** (350 Negative, 300 Neutral, 350 Positive) split into 80% Training, 10% Validation, and 10% Testing sets using stratified sampling.
* **Model Performance:** Fine-tuned `indobenchmark/indobert-base-p1` (AdamW optimizer, `lr=2e-5`, `max_len=128`, 5 epochs), achieving **80.0% test accuracy** (96.5% precision on Positive class; 96.7% recall on Neutral class).
* **Overall Sentiment Distribution (6,456 Tweets):**
  * **Neutral:** 45.8% (2,959 tweets)
  * **Negative:** 37.6% (2,426 tweets)
  * **Positive:** 16.6% (1,071 tweets)

## Tech Stack
* **Language:** Python
* **Deep Learning & NLP:** PyTorch, Hugging Face Transformers (IndoBERT), Scikit-Learn
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn, WordCloud
* **Dashboard & Environment:** Streamlit, Google Colab (Tesla T4 GPU)

## Dashboard & Evaluation Preview
<img width="1600" height="827" alt="overview" src="https://github.com/user-attachments/assets/df862948-9702-4d57-b53c-b84d067945d2" />
<img width="1600" height="827" alt="Noise removal" src="https://github.com/user-attachments/assets/90d6c8a4-324e-44c1-be6c-a2428ade9eea" />
<img width="1600" height="827" alt="labeled" src="https://github.com/user-attachments/assets/e47770ad-c54a-4a28-8c31-75b3b7b124e0" />
<img width="1600" height="826" alt="import 2" src="https://github.com/user-attachments/assets/4bd44294-b665-4520-97ea-67e444e856de" />
<img width="1600" height="830" alt="home" src="https://github.com/user-attachments/assets/fe951b51-6f51-42ea-9cab-36e5d80c89c7" />
<img width="1600" height="826" alt="hasil 2" src="https://github.com/user-attachments/assets/4c49a2b0-6812-4f90-9c5a-f477a81c4627" />
<img width="1600" height="822" alt="hasil 1" src="https://github.com/user-attachments/assets/1358bae4-4270-456c-9edc-fcbc37b3061c" />
<img width="1600" height="822" alt="finish" src="https://github.com/user-attachments/assets/ad47e407-8309-4f3b-8161-1b83a356efc7" />
<img width="1600" height="829" alt="case folding" src="https://github.com/user-attachments/assets/1adcad21-b5d7-4961-b3b4-de862d95371c" />
