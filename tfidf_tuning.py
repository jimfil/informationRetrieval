import os
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import ParameterGrid

# Import από το δικό σας αρχείο με τις μετρικές
from evaluationMetricsFunctions import precision_at_k, recall, f1_score

def load_docs(folder_path="docs"):
    #Φορτώνει τα έγγραφα από τον καθορισμένο φάκελο.
    docs = {}
    for filename in sorted(os.listdir(folder_path)):
        doc_id = filename
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            docs[doc_id] = file.read()
    return docs

def load_queries(file_path="Queries.txt"):
    #Φορτώνει τα ερωτήματα από το αρχείο.
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            queries[i + 1] = line.strip()
    return queries

def load_relevant(file_path="Relevant.txt"):
    """Φορτώνει τα σχετικά έγγραφα για κάθε ερώτηma."""
    relevant_docs = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            query_id = int(parts[0])
            doc_id = parts[1].zfill(5) # Εξασφαλίζει ότι το ID έχει 5 ψηφία (π.χ. '00090')
            if query_id not in relevant_docs:
                relevant_docs[query_id] = []
            relevant_docs[query_id].append(doc_id)
    return relevant_docs

def evaluate_model(retrieved_ranks, relevant_docs):
    """Αξιολογεί ένα μοντέλο για όλα τα ερωτήματα και επιστρέφει μέσες τιμές."""
    avg_precision_10 = 0
    avg_recall = 0
    avg_f1 = 0
    num_queries = len(retrieved_ranks)

    for query_id, retrieved in retrieved_ranks.items():
        relevant = relevant_docs.get(query_id, [])
        retrieved_ids = [doc_id for doc_id, score in retrieved]
        
        avg_precision_10 += precision_at_k(retrieved_ids, relevant, 10)
        avg_recall += recall(retrieved_ids, relevant)
        avg_f1 += f1_score(retrieved_ids, relevant)

    return {
        "Mean Precision@10": avg_precision_10 / num_queries,
        "Mean Recall": avg_recall / num_queries,
        "Mean F1-Score": avg_f1 / num_queries,
    }

def run_experiment(docs, queries, params):
    """
    Εκτελεί ένα πείραμα με συγκεκριμένες παραμέτρους TfidfVectorizer.
    Χρονομετρεί τη δεικτοδότηση και την ανάκτηση.
    """
    doc_ids = list(docs.keys())
    doc_contents = list(docs.values())
    
    # Briskoume weight apo document
    start_time_indexing = time.time()
    vectorizer = TfidfVectorizer(**params)   # gia na mhn grafoume kathe fora TfidfVectorizer(**params)
    doc_vectors = vectorizer.fit_transform(doc_contents) #pairnoume tf kai idf apo docs
    end_time_indexing = time.time()
    indexing_time = end_time_indexing - start_time_indexing

    # --- Ανάκτηση (Retrieval) ---
    all_query_ranks = {}
    start_time_retrieval = time.time()
    for query_id, query_text in queries.items(): # query_text -> string oxi lista !!!
        query_vector = vectorizer.transform([query_text])  # pairnoume 
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Ταξινόμηση των αποτελεσμάτων
        ranked_indices = np.argsort(-similarities)
        ranked_docs = [(doc_ids[i], similarities[i]) for i in ranked_indices]
        all_query_ranks[query_id] = ranked_docs
    end_time_retrieval = time.time()
    retrieval_time = end_time_retrieval - start_time_retrieval

    return all_query_ranks, indexing_time, retrieval_time



docs = load_docs()
queries = load_queries()
relevant_docs = load_relevant()


param_grid = {
    'ngram_range': [(1, 1), (1, 2)], # plhthos syndiasmwn leksewn 
    'sublinear_tf': [True, False], # 1 + log(tf) H  tf 
    'min_df': [1, 2, 5], # h elaxistes fores pou prepei na emfanistei gia na symperilifthei
    'norm': ['l1', 'l2', None] # eukleidia kanonikopoihsh H manhatan H kamia
}

results = []

print("Running experiments with TfidfVectorizer...")
param_combinations = list(ParameterGrid(param_grid))

for i, params in enumerate(param_combinations):
    print(f"Running combination {i+1}/{len(param_combinations)}: {params}")
    
    # Εκτέλεση πειράματος
    ranked_docs_sklearn, indexing_time, retrieval_time = run_experiment(docs, queries, params)
    # Αξιολόγηση
    metrics = evaluate_model(ranked_docs_sklearn, relevant_docs)
    
    results.append({
        'params': params,
        'metrics': metrics,
        'indexing_time': indexing_time,
        'retrieval_time': retrieval_time
    })

# 4. Εύρεση και Εκτύπωση Καλύτερου Μοντέλου
best_model = max(results, key=lambda x: x['metrics']['Mean F1-Score'])

print("\n--- Best Model Found ---")
print(f"Parameters: {best_model['params']}")
print("Metrics:")
for metric, value in best_model['metrics'].items():
    print(f"  {metric}: {value:.4f}")
print(f"Indexing Time: {best_model['indexing_time']:.4f} seconds")
print(f"Retrieval Time: {best_model['retrieval_time']:.4f} seconds")

# 5. Σύγκριση με τις δικές σας υλοποιήσεις (Ερώτημα 2)
print("\n--- Comparison with Custom Implementations ---")

# Φόρτωση αποτελεσμάτων από το findDocumentRanks1.py
try:
    start_custom_indexing = time.time()
    os.system('py analyshEurethriou1.py') # Θα μπορούσαμε να τα τρέξουμε έτσι
    
    
    # ... αλλά ας υποθέσουμε έναν χρόνο για την ανάλυση
    custom_indexing_time = time.time() - start_custom_indexing
    
    
    start_custom_retrieval = time.time()
    os.system('py analyshErwthsewn1.py')
    os.system('py findDocumentRanks1.py')
    
    custom_retrieval_time = time.time() - start_custom_retrieval
    
    with open('sortedRelevant.json', 'r') as f:
        custom_ranks = json.load(f)
    custom_metrics = evaluate_model(custom_ranks, relevant_docs)
    
    print("\nMetrics for Custom TF-IDF Model (Ερώτημα 2):")
    for metric, value in custom_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Estimated Indexing Time: {custom_indexing_time:.4f} seconds")
    print(f"Estimated Retrieval Time: {custom_retrieval_time:.4f} seconds")

except :
    print("\nCould not find 'sortedRelevant.json'. Skipping comparison with custom model.")
print("\n--- Comparison with Custom Implementations ---")

# Φόρτωση αποτελεσμάτων από το findDocumentRanks1.py
try:
    
    start_custom_indexing = time.time()
    os.system('py analyshEurethriou2.py') # Θα μπορούσαμε να τα τρέξουμε έτσι
    
    custom_indexing_time = time.time() - start_custom_indexing

    
    start_custom_retrieval = time.time()
    os.system('py analyshErwthsewn2.py')
    os.system('py findDocumentRanks2.py')

    custom_retrieval_time = time.time() - start_custom_retrieval
    with open('sortedRelevant2.json', 'r') as f:
            custom_ranks = json.load(f)


    custom_metrics = evaluate_model(custom_ranks, relevant_docs)
    
    print("\nMetrics for Custom TF-IDF Model (Ερώτημα 2):")
    for metric, value in custom_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Estimated Indexing Time: {custom_indexing_time:.4f} seconds")
    print(f"Estimated Retrieval Time: {custom_retrieval_time:.4f} seconds")

except FileNotFoundError:
    print("\nCould not find 'sortedRelevant.json'. Skipping comparison with custom model.")