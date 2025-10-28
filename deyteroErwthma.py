from analyshErwthsewn import queryVector
from analyshEurethriou import tfidf_vectors
from math import sqrt  #gia ypologismo Cosine Simularity

def cosine_similarity(wQuaries, wDocuments):
    
    common_terms = set(wQuaries.keys()) & set(wDocuments.keys())                    # lambanoume koina kleidia apo DOCUMENT kai QUERIES
    numerator = sum(wQuaries[t] * wDocuments[t] for t in common_terms)              # Arithmiths

    wq = sum(v * v for v in wQuaries.values())                                      # Ypologismos paronwmasth ||A||*||D||
    wd = sum(v * v for v in wDocuments.values())                                    # Riza(Σ wq^2 * Σ wd^2)
    denominator = sqrt(wq * wd)                                               

    return (numerator / denominator) if denominator != 0 else 0

answer = {}
def findDoc(query):
    for filename in tfidf_vectors.keys():
        answer[filename] = cosine_similarity(queryVector[query],tfidf_vectors[filename])
    sortd = sorted(answer.items(), key=lambda x: x[1], reverse=True)                # ta apotelesmata einai konta metaksi tous 0.59 to prwto me 0.57 to deutero (fysiologiko nmzs)
    return sortd                                                                     
