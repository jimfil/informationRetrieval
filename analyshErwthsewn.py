queryVector = {}
idfWords = {}

with open("dict.txt", "r") as file:  
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) < 2:                              # skip 1h grammh (kai oses einai kenes H ligotera apo 2 stoixeia)
            continue

        term = parts[0].strip()                         # pairnoume term
        idf = float(parts[1].strip())                   # pairnoume to idf  
        idfWords[term] = idf                            # dhmiourgoume to leksiko mas


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
            queryVector[i][term] = tfidfValue           # 1o leksiko 1h erwthsh, 2o leksiko 2h erwthsh klp
        i += 1
                                                




        

    


