import json 

queryVector = {}
idfWords = {}

with open("inverted_index.json", "r") as f:
    data = json.load(f)

for key in data.keys():
        idfWords[key] = data[key].pop()

        
with open("Queries.txt", "r") as file: 
    i = 1
    for line in file:
        words = line.strip().lower().split()
        tf_query = {}
        for term in words:
            tf_query[term] = tf_query.get(term, 0) + 1
        for term in tf_query:
            tf_query[term] /= len(words)
            if term in idfWords: idftemp =idfWords[term]# an DEN yparxei auth h leksh sto leksiko mas den tha thn psaksoume ara idfvalue=0 
            else:  idftemp = 0 
            tfidfValue = tf_query[term] * idftemp       # ypologismos tfidfvalues
            if i not in queryVector:                    # an den yparxei dict gia auto to arxeio dhmiourghse to 
                queryVector[i] = {}
            queryVector[i][term] = tfidfValue           # nested dictionary: query Number -> word -> tfidf value
        i += 1
                                                
with open("queryVector.json", "w") as f: json.dump(queryVector, f, indent=4)



        

    


