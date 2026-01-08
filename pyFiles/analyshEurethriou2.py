import json
from math import sqrt

termWeight_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}

with open("textFiles/inverted_index.json", "r") as f:
    data = json.load(f)
logValu = {}
fileDict = {}

for key in data.keys():                                     # gia kathe leksi
    logValu[key] = data[key].pop()
    for item in data[key]:                                  # pare kathe arxeio pou yparxei h leksh auth
        mult = (item[1]* logValu[key])                      # ypologise to tf * log
        if item[0] not in fileDict:                         # an den yparxei dict gia auto to arxeio dhmiourghse to 
            fileDict[item[0]] = {} 
        fileDict[item[0]][key] = mult

                                                            # pleon exoume kathe arxeio me oti TERM exei

for filename in fileDict.keys():                                 # gia kathe arxeio
    temp = 0  
    for term2 in fileDict[filename].keys():
        temp += (fileDict[filename][term2])* (fileDict[filename][term2])
    denom = sqrt(temp)
    for term in fileDict[filename].keys():                       # gia kathe leksi
        num = fileDict[filename][term] 
        docWeight = num / denom          
        
        if filename not in termWeight_vectors:           # an den yparxei dict gia auto to arxeio dhmiourghse to 
            termWeight_vectors[filename] = {}
        termWeight_vectors[filename][term] = docWeight       # nested dictionary: filename -> word -> tfidf value 
                                                        

with open("textFiles/tfidfVectors2.json", "w") as f: json.dump(termWeight_vectors, f, indent=4)

