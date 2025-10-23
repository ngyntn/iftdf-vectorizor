
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

from profile.profile_vectorizer import profile_update_job
from tfidf.tfidf_builder import query_articles_from_db, vectorize_articles, index_to_es, index_articles_job

app = Flask(__name__)

scheduler = BackgroundScheduler()

scheduler.add_job(index_articles_job, 'interval', seconds=10, id='index_job')
scheduler.add_job(profile_update_job,'interval', seconds=20, id='profile_job')


scheduler.start()

@app.route("/")
def home():
    return "Hello Flask"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
