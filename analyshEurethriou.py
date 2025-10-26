tfidf_vectors = {}   # filename -> {term: tfidf} , {term2:tfidf}


with open("dict.txt", "r") as file:  
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) < 2:                              # skip 1h grammh (kai oses einai kenes H ligotera apo 2 stoixeia)
            continue

        term = parts[0].strip()                         # pairnoume term
        idf = float(parts[1].strip())                   # pairnoume to idf  
                                                        
        for pair in parts[2:]:                          # gia kathe lista meta to idf bgale tis ()    
            pair = pair.strip().strip("() ")            
            filename, tf = pair.split(",")              # xwrise to onoma tou document me to term frequency
            tf = float(tf)                              # float (tf) 
            tfidf = tf * idf                            # bres EPITELOUS to tfidf

            if filename not in tfidf_vectors:           # an den yparxei dict gia auto to arxeio dhmiourghse to 
                tfidf_vectors[filename] = {}
            tfidf_vectors[filename][term] = tfidf       # nested dictionary: filename -> word -> tfidf value 
                                                        


