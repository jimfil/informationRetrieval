import json 

queryVector = {}

with open("textFiles/inverted_index.json", "r") as f:
    data = json.load(f)
logValu = {}
for key in data.keys():
    logValu[key] = data[key].pop()

tf_query ={}
with open("textFiles/Queries.txt", "r") as file: 
    i = 1
    for line in file:
        words = line.strip().lower().split()
        if i not in tf_query: tf_query[i] = {}
        for term in words:
            tf_query[i][term] = tf_query[i].get(term, 0) + 1
        for term in tf_query[i]:
            tf_query[i][term] /= len(words)
        i += 1

for i in tf_query.keys():
    max_tf = max(tf_query[i].values())
    for term in tf_query[i].keys(): 
        qw = (0.5 + 0.5 * tf_query[i][term] / max_tf) * logValu.get(term, 0.0)
        if i not in queryVector: queryVector[i] = {}
        queryVector[i][term]= qw


with open("textFiles/queryVector2.json", "w") as f: json.dump(queryVector, f, indent=4)
  
