
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
from elasticsearch import Elasticsearch
from datetime import datetime

from config.db import close_db, connect_db
from preprocess.eng_processor import clean_text



def query_articles_from_db():
    connection = connect_db()

    cursor = connection.cursor()
    print("start query")
    cursor.execute("SELECT id, title, content FROM articles WHERE id >= 10 AND moderation_status='approved'")
    rows = cursor.fetchall()
    articles = [{"id": row[0], "title": row[1], "content": row[1] + " " + row[2]} for row in rows]

    close_db(connection)
    print(f"Queried {len(articles)} articles from DB.")
    return articles

# Vectorization và Reduction (Tăng max_features cho vocab tiếng Anh phong phú)
def vectorize_articles(articles):
    cleaned_texts = [clean_text(article['content']) for article in articles]
    print("cleaned texts ", cleaned_texts)

    # TF-IDF với giới hạn features cao hơn
    vectorizer = TfidfVectorizer(max_features=8000, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    vocab_size = tfidf_matrix.shape[1]

    # Giảm chiều (giữ 1000 dims cho ES)
    n_components = min(1000, max(50, vocab_size))
    svd = TruncatedSVD(n_components=n_components)
    reduced_vectors = svd.fit_transform(tfidf_matrix)

    # Lưu models
    joblib.dump(vectorizer, 'model/tfidf_model.pkl') # Đường dẫn dựa trên CWD (vị trí file thực thi)
    joblib.dump(svd, 'model/svd_model.pkl')

    # Gán vector cho articles
    for i, article in enumerate(articles):
        article['vector'] = reduced_vectors[i].tolist()
        article['dims'] = reduced_vectors.shape[1]

    print(
        f"Vectorized {len(articles)} articles (reduced to 1000 dims). Vocab size: {len(vectorizer.get_feature_names_out())}")
    return articles



# Index vào Elasticsearch
def index_to_es(articles):
    if not articles:
        print("No articles to index")

    dims = articles[0]['dims']
    es = Elasticsearch(['http://localhost:9200'])

    if es.indices.exists(index="articles"):
        es.indices.delete(index="articles")

    # Tạo index nếu chưa có
    if not es.indices.exists(index="articles"):
        es.indices.create(
            index="articles",
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "integer"},
                        "title": {"type": "text"},
                        "vector": {"type": "dense_vector", "dims": dims}
                    }
                }
            }
        )
        print("ES index 'articles' created.")

    # Xóa data cũ và index mới
    es.delete_by_query(index="articles", body={"query": {"match_all": {}}})
    for article in articles:
        doc = {
            "id": article['id'],
            "title": article['title'],
            "vector": article['vector']
        }
        es.index(index="articles", id=article['id'], body=doc)

    print("Articles indexed to ES.")

def index_articles_job():
    print(f"[{datetime.now()}] Running vectorized article job")
    articles = query_articles_from_db()
    articles = vectorize_articles(articles)
    index_to_es(articles)
    print(f"[{datetime.now()}] Vectorized article job successfully.\n")