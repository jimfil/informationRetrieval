import os
from math import log
import json

tokens = []
filenames = []
repMatrix = []

folder = "docs"
    
for filename in os.listdir(folder):
    with open(folder + "\\" +filename, "r") as file:
        docum = file.read().lower()
    filenames.append(filename)

    tokens.append(docum.split())
   

terms = list(set(word for doc in tokens for word in doc))

inverted_index = {}

for term in terms:                                                                          # Gia kathe diaforetikh leksh se ola ta files
    termInDocuments = []                                                                    # H tokens exei oles tis lekseis apo keimena se listes (listes se lista)
    for i, doc in enumerate(tokens):                                                        # Opote gia kathe lista (keimeno) sthn tokens
        if term in doc:                                                                     # Des ean yparxei h leksh sthn lista
            tfTermInDoc = (doc.count(term)/len(doc))                                        # An nai bres to term Frequency 
            termInDocuments.append([filenames[i], tfTermInDoc])                             # Kai ftiaxe ena list (me to onoma tou arxeiou kai to tf) mesa se ena deutero list px [[00001 , 2] , [00002 , 1]] 
    if len(termInDocuments): idfTerm = log(len(doc)/len(termInDocuments) , 10)              # Ypologise to idf (log(Documents in total / Documents me thn leksi))
    termInDocuments.append(idfTerm)                                                         # Balto sto telos tou deuterou list  px [[00001 , 2] , [00002 , 1] , 0.173]

    inverted_index[term] = termInDocuments                                                  # Olo auto einai ena dictionary me kleidi thn leksh 


json.dump(inverted_index, open("dict.json", "w"), indent=4)                                 # Dhmiourgoume ena json arxeio


