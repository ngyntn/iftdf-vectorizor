Here is a professional, comprehensive, and clear **README.md** tailored to your project. It includes all the requested details, icons, and specific instructions for Windows.

-----

# 🚀 AI Recommendation Service

> **An intelligent article recommendation system powered by TF-IDF, SVD, and Elasticsearch.**

   

## 📖 Introduction

This project is a high-performance **Microservice** designed to provide real-time search and personalized article recommendations for a blog platform.

It utilizes **Natural Language Processing (NLP)** techniques to vectorise article content and user reading history. By combining **content relevance** (via Cosine Similarity) and **freshness** (Time Decay), the system delivers highly accurate and up-to-date suggestions.

## 🛠 Tech Stack

  * **Core:** Python 3.13, Flask (API Framework)
  * **Machine Learning:**  Scikit-learn (TF-IDF, TruncatedSVD), NumPy, NLTK
  * **Search Engine:**  Elasticsearch 8.11.0 (Vector Search/KNN)
  * **Task Queue:** Celery + Redis (Broker)
  * **Database:**  MySQL
  * **Scheduler:**  APScheduler (Background jobs)

## ✨ Key Features

1.  **KNN Semantic Search:** Search for articles based on meaning rather than just keywords using Vector Search.
2.  **Hybrid Recommendation:** Suggests articles based on a user's reading history, balancing **Relevance (60%)** and **Freshness (40%)**.
3.  **Automated Indexing:**
      * **Full Indexing (Daily):** Re-trains models and re-indexes all data.
      * **Incremental Indexing (Hourly):** Processes new articles continuously without downtime.
4.  **Distributed Processing:** Uses Celery workers to handle heavy text cleaning and indexing tasks asynchronously.

-----

## ⚙️ Installation & Setup

### 1\. Prerequisites (Infrastructure)

Before running the Python code, you must have the following infrastructure running. **Docker** is recommended.

  * **MySQL Database:** Ensure you have a database named `web_blog_db` and the `articles` table exists.
  * **Redis:** Required for Celery message brokering.
  * **Elasticsearch (Strict Version):** You **must** use version `8.11.0`. Other versions may cause compatibility issues with the client library.

**Docker Quick Start (Recommended):**

```bash
# Run Redis
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Run Elasticsearch 8.11.0 (Single node for dev)
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### 2\. Python Environment Setup

The project requires **Python 3.13**. Follow these steps to set up the environment from scratch.

**Step A: Clone and Enter Project**

```bash
git clone https://github.com/ngyntn/iftdf-vectorizor.git
cd iftdf-vectorizor
```

**Step B: Create Virtual Environment**
If you don't have a `.venv` folder yet, create one to isolate dependencies:

```bash
# Windows
python -m venv .venv
```

**Step C: Activate Environment**

```bash
# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Step D: Install Dependencies**

```bash
pip install -r requirements.txt
```

*(Note: The application will automatically download necessary NLTK data like stopwords and wordnet on the first run).*

### 3\. Environment Variables

Create a file named `.env` in the root directory. Copy the content below and update `DB_HOST` or credentials if necessary.

```ini
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=3307
DB_USER=admin
DB_PASS=webblogpassword
DB_NAME=web_blog_db

# Search Engine & Broker
ES_URL=http://localhost:9200
REDIS_URL=redis://localhost:6379
```

-----

## 🚀 Running the Application (Windows)

To handle the workload efficiently, we run the **Flask API** and **4 Celery Workers** simultaneously. On Windows, you should open **5 separate Terminal Tabs** (or Command Prompt windows).

> **⚠️ Important Note for Windows Users:**
> Python on Windows does not support `fork`. Therefore, when running Celery, you usually need to use the `solo` or `threads` pool, or simply run multiple worker instances in separate tabs as shown below.

### 🔲 Tab 1: Run the Flask API & Scheduler

This starts the web server and the background scheduler (APScheduler).

```bash
# Ensure .venv is activated
python app.py
```

*Server will start at `http://127.0.0.1:5000`*

### 🔲 Tab 2: Celery Worker 1

Handles background tasks (Text cleaning, Indexing).

```bash
# Ensure .venv is activated
celery -A worker.celery worker --loglevel=info --pool=solo --hostname=worker1@%h
```

### 🔲 Tab 3: Celery Worker 2

```bash
# Ensure .venv is activated
celery -A worker.celery worker --loglevel=info --pool=solo --hostname=worker2@%h
```

### 🔲 Tab 4: Celery Worker 3

```bash
# Ensure .venv is activated
celery -A worker.celery worker --loglevel=info --pool=solo --hostname=worker3@%h
```

### 🔲 Tab 5: Celery Worker 4

```bash
# Ensure .venv is activated
celery -A worker.celery worker --loglevel=info --pool=solo --hostname=worker4@%h
```

-----

## 📡 API Endpoints

Once the app is running, you can interact with it via REST API.

### 1\. Semantic Search

**GET** `/articles/search/knn?key=machine+learning&page=1&size=10`

  * **key**: The search query string.
  * **page**: Page number.
  * **size**: Number of results per page.

### 2\. Get Recommendations

**POST** `/articles/recommend`

**Body (JSON):**

```json
{
    "user_id": 8,
    "read_ids": [101, 102, 103]
}
```

  * **user\_id**: ID of the user requesting recommendations.
  * **read\_ids**: List of article IDs the user has already read (to exclude them from results).

-----

## 📂 Project Structure

```text
├── config/
│   └── db.py              # MySQL Connection logic
├── knn/
│   ├── query.py           # KNN Search Logic
│   └── recommend.py       # Recommendation Logic (Relevance + Freshness)
├── model/                 # Stores .pkl files (TF-IDF, SVD)
├── preprocess/
│   └── eng_processor.py   # NLTK Text Cleaning
├── tfidf/
│   └── tfidf_builder.py   # Job definitions for indexing
├── app.py                 # Main Flask Entry point + Scheduler
├── worker.py              # Celery Worker Tasks
├── requirements.txt       # Python Dependencies
├── .env                   # Environment Variables
└── README.md              # Documentation
```