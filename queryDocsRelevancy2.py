import json

with open("sortedRelevant.json", "r") as f:
    sortedRelevant = json.load(f)

name = 0 
while 1:
    print("Select the number of the query (1-20) you want to search:")
    try: name = int(input())
    except: continue
    if name in range(1,21): break

for i in range(5):

    print(f"{i+1}. Document number: {sortedRelevant[str(name)][i]}")