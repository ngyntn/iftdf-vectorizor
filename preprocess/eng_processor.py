
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# Preprocessing - Làm sạch text
def get_wordnet_pos(word):
    # Lấy POS tag cho lemmatization (tối ưu tiếng Anh).
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean_text(text):
    if not text: return 0
    words = nltk.word_tokenize(text.lower())  # Tokenize chuẩn
    words = [lemmatizer.lemmatize(stemmer.stem(word), get_wordnet_pos(word))
             for word in words
             if word not in stop_words and len(word) > 2 and word.isalpha()]  # Chỉ chữ cái, bỏ số/ký tự
    return ' '.join(words)