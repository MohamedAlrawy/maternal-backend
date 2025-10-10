from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(query, qa_list):
    if not qa_list:
        return None

    questions = [qa.question for qa in qa_list]
    
    vectorizer = TfidfVectorizer().fit_transform([query] + questions)
    print(vectorizer)
    print(vectorizer[0:1])
    print(vectorizer[1:])
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    print("lsdjfgklsdfhglkdsgklsdjglskdgjkl;sdfgjl;sdgjl;kxdjgkldjgkl;sdjgm")
    print(similarities)

    best_index = similarities.argmax()
    print("best_index")
    print(best_index)
    print(similarities[best_index])
    if similarities[best_index] > 0.1:  # threshold
        print("llllllllllllllllllllllllllllllllllllllllllll")
        print(qa_list[best_index])
        return qa_list[best_index]
    return None
