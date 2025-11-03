import json
from math import sqrt  #gia ypologismo Cosine Simularity

with open("queryVector.json", "r") as f:
    queryVector = json.load(f)
with open("termWeight_vectors.json", "r") as f:
    tfidf_vectors = json.load(f)

def cosine_similarity(wQuaries, wDocuments):
    
    common_terms = set(wQuaries.keys()) & set(wDocuments.keys())                    # lambanoume koina kleidia apo DOCUMENT kai QUERIES
    numerator = sum(wQuaries[t] * wDocuments[t] for t in common_terms)              # Arithmiths

    wq = sum(v * v for v in wQuaries.values())                                      # Ypologismos paronwmasth ||A||*||D||
    wd = sum(v * v for v in wDocuments.values())                                    # Riza(Σ wq^2 * Σ wd^2)
    denominator = sqrt(wq * wd)                                               
    print(numerator/denominator)
    return (numerator / denominator) if denominator != 0 else 0

def findDoc(query):
    answer = {}
    for filename in tfidf_vectors.keys():
        answer[filename] = cosine_similarity(queryVector[str(query)],tfidf_vectors[filename])
    sortd = sorted(answer.items(), key=lambda x: x[1], reverse=True)                # ta apotelesmata einai konta metaksi tous 0.59 to prwto me 0.57 to deutero (fysiologiko nmzs)
    return sortd                                                                     

sortedDocum = {}
for i in range(1,21):
    if i not in tfidf_vectors:
        sortedDocum[i] = []
    lista = findDoc(i)
    for item in lista:    
        sortedDocum[i].append(item[0])


with open("sortedRelevant.json", "w") as f: json.dump(sortedDocum, f, indent=4)
