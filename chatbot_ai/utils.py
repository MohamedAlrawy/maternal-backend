from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(query, qa_list):
    if not qa_list:
        return None

    questions = [qa.question for qa in qa_list]
    vectorizer = TfidfVectorizer().fit_transform([query] + questions)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    best_index = similarities.argmax()
    if similarities[best_index] > 0.5:  # threshold
        return qa_list[best_index]
    return None
