import json
from math import sqrt,log  #gia ypologismo Cosine Simularity

def cls_analyshEurethriou():
    tfidf_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}

    with open("textFiles/inverted_index.json", "r") as f:
        data = json.load(f)

    for key in data.keys():
            idfValue = log(data[key].pop(),10)

            for item in data[key]:
                filename = item[0]
                tfValue = item[1]                           
                tfidf = tfValue * idfValue                    

                if filename not in tfidf_vectors:           # an den yparxei dict gia auto to arxeio dhmiourghse to 
                    tfidf_vectors[filename] = {}
                tfidf_vectors[filename][key] = tfidf       # nested dictionary: filename -> word -> tfidf value 
                                                            
    with open("textFiles/tfidfVectors.json", "w") as f: json.dump(tfidf_vectors, f, indent=4)

def cls_analyshErwthsewn():
    queryVector = {}
    idfWords = {}

    with open("textFiles/inverted_index.json", "r") as f:
        data = json.load(f)

        for key in data.keys():
            idfWords[key] = log(data[key].pop(),10)

                
        with open("textFiles/Queries.txt", "r") as file: 
            i = 1
            for line in file:
                words = line.strip().lower().split()
                tf_query = {}
                for term in words:
                    tf_query[term] = tf_query.get(term, 0) + 1
                for term in tf_query.keys():
                    tf_query[term] /= len(words)
                    tfidfValue = (tf_query[term]) * idfWords.get(term, 0.0)      # ypologismos tfidfvalues
                    if i not in queryVector:                    # an den yparxei dict gia auto to arxeio dhmiourghse to 
                        queryVector[i] = {}
                    queryVector[i][term] = tfidfValue           # nested dictionary: query Number -> word -> tfidf value
                i += 1
                                                        
    with open("textFiles/queryVector.json", "w") as f: json.dump(queryVector, f, indent=4)

def cls_findDocumentRanks():
    with open("textFiles/queryVector.json", "r") as f:
        queryVector = json.load(f)
    with open("textFiles/tfidfVectors.json", "r") as f:
        tfidf_vectors = json.load(f)

    denom_values_query = {}
    denom_values_doc = {}

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


    with open("textFiles/sortedRelevant1.json", "w") as f: json.dump(sortedDocum, f, indent=4)


def cls_printRelevancy():
    with open("textFiles/sortedRelevant1.json", "r") as f:
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
    if 'q' in input("Press Enter to start TF-IDF Tuning Process or q to quit: ").lower():
        exit()
    print("Creating TF-IDF Vectors for Documents and Queries")
    cls_analyshEurethriou()
    cls_analyshErwthsewn()
    print("Created Succesfully")
    print("Finding Document Ranks for each Query")
    cls_findDocumentRanks()
    cls_printRelevancy()