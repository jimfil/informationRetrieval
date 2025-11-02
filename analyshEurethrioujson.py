import json
tfidf_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}
with open("dict.json", "r") as f:
    data = json.load(f)

for key in data.keys():
    idfvalues = float(data[key].pop())

    for item in data[key]:
        filename = item[0]
        tf = float(item[1])
        tfidfValue = idfvalues * tf

        if filename not in tfidf_vectors:
            tfidf_vectors[filename] = {}
        tfidf_vectors[filename][key] = tfidfValue

json.dump(tfidf_vectors, open("tfidf_vectors.json", "w"), indent=3)   
