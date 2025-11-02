import json

tfidf_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}

with open("inverted_index.json", "r") as f:
    data = json.load(f)

for key in data.keys():
        idfValue = data[key].pop()

        for item in data[key]:
            filename = item[0]
            tfValue = item[1]                           
            tfidf = tfValue * idfValue                    

            if filename not in tfidf_vectors:           # an den yparxei dict gia auto to arxeio dhmiourghse to 
                tfidf_vectors[filename] = {}
            tfidf_vectors[filename][key] = tfidf       # nested dictionary: filename -> word -> tfidf value 
                                                        

with open("tfidf_vectors.json", "w") as f: json.dump(tfidf_vectors, f, indent=4)

