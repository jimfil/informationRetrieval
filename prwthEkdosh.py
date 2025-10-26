import os
tokens = []
filenames = []
folder = "docs"
    
for filename in os.listdir(folder):
    with open(folder + "\\" +filename, "r") as file:
        docum = file.read()
    filenames.append(filename)
    tokens.append(docum.lower().split())
    

terms = list(set(word for doc in tokens for word in doc))

inverted_index = {}

for term in terms:
    documents = []
    for i, doc in enumerate(tokens):
        if term in doc:
            documents.append(filenames[i])
    inverted_index[term] = documents


with open("dictPrwthEkdosh.txt", "w") as file:
    for term in inverted_index.keys():
            file.write("\n"+ term)
            for document in inverted_index[term]:
                file.write("\t"+ document + " ,")


        