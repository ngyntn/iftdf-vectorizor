## ✨ Introduction
This backend service provides **Contextual Search** and **Personalized Recommendation** capabilities for an article platform. It leverages NLP and vectorization techniques to store article content and user preferences, enabling highly relevant results delivered via a Flask API.

## Key Technologies
- **Framework:** Flask  
- **Scheduling:** APScheduler (for background jobs)  
- **Vectorization:** Scikit-learn (TF-IDF and Truncated SVD to reduce vectors to 1000 dimensions)  
- **Search/Storage:** Elasticsearch (used as a Vector Database for KNN search and profile storage)  
- **Data Source:** Relational Database (e.g., MySQL/PostgreSQL) for articles and user interactions  

## Mechanism of Operation
The system operates through two primary pipelines:

### 1️⃣ Article Indexing Pipeline (`tfidf_builder.py`)
- Articles are cleaned and vectorized using a trained TF-IDF model.  
- Dimensionality is reduced via Truncated SVD to 1000 dimensions.  
- Resulting article vectors are stored in the **Elasticsearch `articles` index**.  
- Jobs run:
  - **Daily:** Full re-index  
  - **Hourly:** Incremental index 

### 2️⃣ User Profile Pipeline (`profile_vectorizer.py`)
- User profiles are built using a **weighted average** of article vectors they interacted with in the last 90 days.  
- Interaction scores (Click, Read, Like, etc.) are weighted with a **time decay factor** (0.9 decay per week) ⏱️ to prioritize recent activity.  
- Final **User Profile Vector** is stored in the **Elasticsearch `user_profiles` index**.  
- Profile update jobs run **hourly** for active users.  

## API Endpoints
The service exposes two main functionalities:

### 1. Content Search
- **Endpoint:** `/articles/search/knn`  
- **Method:** `GET`  
- **Functionality:** Performs a KNN search on article vectors based on the user's input query.  

### 2. Personalized Recommendations
- **Endpoint:** `/articles/recommend`  
- **Method:** `POST`  
- **Functionality:** Fetches the user's profile vector and finds nearest neighbor articles based on vector similarity.  

