import json
from math import sqrt 


def analyshErwthsewn():
    queryVector = {}

    with open("textFiles/inverted_index.json", "r") as f:
        data = json.load(f)
    logValu = {}
    for key in data.keys():
        logValu[key] = data[key].pop()

    tf_query ={}
    with open("textFiles/Queries.txt", "r") as file: 
        i = 1
        for line in file:
            words = line.strip().lower().split()
            if i not in tf_query: tf_query[i] = {}
            for term in words:
                tf_query[i][term] = tf_query[i].get(term, 0) + 1
            for term in tf_query[i]:
                tf_query[i][term] /= len(words)
            i += 1

    for i in tf_query.keys():
        max_tf = max(tf_query[i].values())
        for term in tf_query[i].keys(): 
            qw = (0.5 + 0.5 * tf_query[i][term] / max_tf) * logValu.get(term, 0.0)
            if i not in queryVector: queryVector[i] = {}
            queryVector[i][term]= qw

    
    with open("textFiles/queryVector2.json", "w") as f: json.dump(queryVector, f, indent=4)

def analyshEurethriou():
    tfidf_vectors = {}

    with open("textFiles/inverted_index.json", "r") as f:
        data = json.load(f)

    for key in data.keys():
        logValue = data[key].pop()

        for item in data[key]:
            filename = item[0]
            tfValue = item[1]
            tfidf = (0.5 + 0.5 * tfValue / max([item[1] for item in data[key]])) * logValue

            if filename not in tfidf_vectors:
                tfidf_vectors[filename] = {}
            tfidf_vectors[filename][key] = tfidf

    with open("textFiles/tfidfVectors2.json", "w") as f: json.dump(tfidf_vectors, f, indent=4)

def findDocumentRanks():
    with open("textFiles/queryVector2.json", "r") as f:
        queryVector = json.load(f)
    with open("textFiles/tfidfVectors2.json", "r") as f:
        tfidf_vectors = json.load(f)

    denom_values_query = {}
    denom_values_doc = {}

    def calculate_denom_values():                   # optimization gia na min ypologizontai kathe fora 
        '''
        Calculate and store the denominator values for all documents and queries.
        '''
        for filename in tfidf_vectors.keys():               # Riza( Σ wd^2)
            denom_values_doc[filename] = 0
            for v in tfidf_vectors[filename].values():
                denom_values_doc[filename] += v * v
            denom_values_doc[filename] = sqrt(denom_values_doc[filename])

        for qname in queryVector.keys():                    # Riza(Σ wq^2)
            denom_values_query[qname] = 0
            for v in queryVector[qname].values():
                denom_values_query[qname] += v * v
            denom_values_query[qname] = sqrt(denom_values_query[qname])

    calculate_denom_values()

    def cosine_similarity(wQuaries, wDocuments,queryName, docName):
        '''Calculate cosine similarity between query and document vectors.'''
        common_terms = set(wQuaries.keys()) & set(wDocuments.keys())                    # lambanoume koina kleidia apo DOCUMENT kai QUERIES
        numerator = sum(wQuaries[t] * wDocuments[t] for t in common_terms)              # Arithmiths

        wq = denom_values_query[queryName]                                 
        wd = denom_values_doc[docName]                                    
        denominator = wq * wd                                               # Ypologismos paronwmasth ||A||*||D||
        return (numerator / denominator) if denominator != 0 else 0

    def findDoc(query):
        answer = {}
        for filename in tfidf_vectors.keys():
            answer[filename] = cosine_similarity(queryVector[str(query)],tfidf_vectors[filename],str(query),filename)
        sortd = sorted(answer.items(), key=lambda x: x[1], reverse=True)              
        return sortd    
    
    sortedDocum = {}
    for i in range(1,21):
        if i not in sortedDocum:
            sortedDocum[i] = []
        lista = findDoc(i)
        for item in lista:    
            sortedDocum[i].append(item)

    with open("textFiles/sortedRelevant2.json", "w") as f: json.dump(sortedDocum, f, indent=4)


def printRelevancy():
    with open("textFiles/sortedRelevant2.json", "r") as f:
        sortedRelevant = json.load(f)

    name = 0 
    while 1:
        print("Select the number of the query (1-20) you want to search:")
        try: name = int(input())
        except: continue
        if name in range(1,21): break

    for i in range(5):

        print(f"{i+1}. Document number: {sortedRelevant[str(name)][i]}")

if __name__ == "__main__":
    input("Press Enter to start TF-IDF Tuning Process  or q to quit: ")
    if 'q' in input().lower():
        exit()
    print("Creating TF-IDF Vectors for Documents and Queries")
    analyshEurethriou()
    analyshErwthsewn()
    print("Created Succesfully")
    print("Finding Document Ranks for each Query")
    findDocumentRanks()
    printRelevancy()
