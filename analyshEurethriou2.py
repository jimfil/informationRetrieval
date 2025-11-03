import json
from math import sqrt

termWeight_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}

with open("inverted_index.json", "r") as f:
    data = json.load(f)
logValu = {}
denom = {}
for key in data.keys():
    tfValue = 0
    logValu[key] = data[key].pop()
    for item in data[key]:
        mult = (item[1]* logValu[key])
        tfValue += (mult * mult)  

    denom[key] = sqrt(tfValue)

for key in data.keys():
             
    for item in data[key]:
        filename = item[0]
        tfValue = item[1]               
        num = tfValue * logValu[key] 
        tw = num / denom[key]  
        if filename not in termWeight_vectors:           # an den yparxei dict gia auto to arxeio dhmiourghse to 
            termWeight_vectors[filename] = {}
        termWeight_vectors[filename][key] = tw       # nested dictionary: filename -> word -> tfidf value 
                                                        

with open("termWeight_vectors.json", "w") as f: json.dump(termWeight_vectors, f, indent=4)

