import json
import time
import os
from createEurethrio import createInvertedIndex
from vectorizer1 import cls_analyshErwthsewn, cls_findDocumentRanks, cls_analyshEurethriou
from vectorizer2 import analyshErwthsewn, findDocumentRanks, analyshEurethriou
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import ParameterGrid
from evaluationMetricsFunctions import plot_precision_recall_curve, precision_at_k, recall, f1_score

def write_line(f, text=""):
    f.write(text + "\n")


def load_docs(folder_path="docs"):
    #Φορτώνει τα έγγραφα από τον καθορισμένο φάκελο.
    docs = {}
    for filename in sorted(os.listdir(folder_path)):
        doc_id = filename
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            docs[doc_id] = file.read()
    return docs

def load_queries(file_path="textFiles/Queries.txt"):
    #Φορτώνει τα ερωτήματα από το αρχείο.
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            queries[i + 1] = line.strip()
    return queries

def load_relevant(file_path="textFiles/Relevant.txt"):
    """Φορτώνει τα σχετικά έγγραφα για κάθε ερώτηma."""
    relevant_docs = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 1
        for line in file:
            query_id = i
            for part in line.split():
                doc_id = part.zfill(5) # Εξασφαλίζει ότι το ID έχει 5 ψηφία (π.χ. '00090')
                if query_id not in relevant_docs:
                    relevant_docs[query_id] = []
                relevant_docs[query_id].append(doc_id)
            i += 1
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
        #plot_precision_recall_curve(retrieved_ids, relevant)
    return {
        "Mean Precision@10": avg_precision_10 / num_queries,
        "Mean Recall": avg_recall / num_queries,
        "Mean F1-Score": avg_f1 / num_queries,
    }

def evaluate_custom_model(retrieved_ranks, relevant_docs):
    """Αξιολογεί τα custom μοντέλα μας για όλα τα ερωτήματα και επιστρέφει μέσες τιμές."""
    avg_precision_10 = 0
    avg_recall = 0
    avg_f1 = 0
    num_queries = len(retrieved_ranks)

    for query_id, retrieved in retrieved_ranks.items():
        
        relevant = relevant_docs.get(int(query_id), [])
        retrieved_ids = [item[0] for item in retrieved]

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


def find_best_model(docs=None, queries=None, relevant_docs=None):
    if docs is None: docs = load_docs()
    if queries is None: queries = load_queries()
    if relevant_docs is None: relevant_docs = load_relevant()

    param_grid = { #2*2*2*2*3 = 48 sunduasmoi
        'ngram_range': [(1, 1), (1, 2)], # plhthos syndiasmwn leksewn 
        'sublinear_tf': [True, False], # 1 + log(tf) H  tf 
        'min_df': [1, 2], # h elaxistes fores pou prepei na emfanistei gia na symperilifthei
        'max_df': [0.8, 1.0], # to megisto pososto twn docs pou mporei na emfanistei h leksi
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
    best_model = max(results, key=lambda x: x['metrics']['Mean Precision@10'])
    with open("textFiles/bestModel.json", "w") as f: json.dump(best_model, f, indent=4)
    return best_model



if __name__ == "__main__":
    docs = load_docs()
    queries = load_queries()
    relevant_docs = load_relevant()

    best_model = find_best_model(docs, queries, relevant_docs)
    """
    # Plotting Precision-Recall curves
    print("\n--- Plotting Precision-Recall Curves ---")
    while True:
        try:
            target_q_id = int(input("Enter Query ID to plot (1-20): "))
            if target_q_id in queries:
                break
            print("Query ID not found. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Best Sklearn Model
    print(f"Plotting for Best Sklearn Model - Query {target_q_id}")
    # Run experiment only for the selected query to save time/resources
    best_ranks, _, _ = run_experiment(docs, {target_q_id: queries[target_q_id]}, best_model['params'])
    if target_q_id in best_ranks:
        ranks = best_ranks[target_q_id]
        relevant = relevant_docs.get(target_q_id, [])
        retrieved_ids = [doc_id for doc_id, score in ranks]
        plot_precision_recall_curve(retrieved_ids, relevant, title=f"Best Sklearn Model - Query {target_q_id}")

    # Custom Model 1
    print(f"Plotting for Custom Model Classic tf-idf - Query {target_q_id}")
    with open("textFiles/sortedRelevant1.json", "r") as f:
        custom_ranks1 = json.load(f)
    if str(target_q_id) in custom_ranks1:
        ranks = custom_ranks1[str(target_q_id)]
        relevant = relevant_docs.get(target_q_id, [])
        retrieved_ids = [item[0] for item in ranks]
        plot_precision_recall_curve(retrieved_ids, relevant, title=f"Custom Model Classic tf-idf - Query {target_q_id}")

    # Custom Model 2
    print(f"Plotting for Custom Model tfc-nfx - Query {target_q_id}")
    with open("textFiles/sortedRelevant2.json", "r") as f:
        custom_ranks2 = json.load(f)
    if str(target_q_id) in custom_ranks2:
        ranks = custom_ranks2[str(target_q_id)]
        relevant = relevant_docs.get(target_q_id, [])
        retrieved_ids = [item[0] for item in ranks]
        plot_precision_recall_curve(retrieved_ids, relevant, title=f"Custom Model tfc-nfx - Query {target_q_id}")
    input("Press Enter to continue...")
    """
    
    with open("textFiles/results.txt", "w", encoding="utf-8") as f:
        write_line(f, "--- Best Model Found ---")
        write_line(f, f"Parameters: {best_model['params']}")
        write_line(f, "Metrics:")
        for metric, value in best_model['metrics'].items():
            write_line(f, f"  {metric}: {value:.4f}")
        write_line(f, f"Indexing Time: {best_model['indexing_time']:.4f} seconds")
        write_line(f, f"Retrieval Time: {best_model['retrieval_time']:.4f} seconds")

        write_line(f, "\n--- Comparison with Custom Implementations ---")

        #1st model
        try:
            start_Indexing = time.time()
            createInvertedIndex()
            time1 = time.time() - start_Indexing

            start_custom_indexing = time.time()
            cls_analyshEurethriou()
            custom_indexing_time = time.time() - start_custom_indexing + time1

            start_custom_retrieval = time.time()
            
            cls_analyshErwthsewn()
            cls_findDocumentRanks()
            custom_retrieval_time = time.time() - start_custom_retrieval

            with open(f'textFiles/sortedRelevant{1}.json', 'r') as jf:
                custom_ranks = json.load(jf)

            custom_metrics = evaluate_custom_model(custom_ranks, relevant_docs)

            write_line(f, f"\nMetrics for Classic Tf-idf Model:")
            for metric, value in custom_metrics.items():
                write_line(f, f"  {metric}: {value:.4f}")
            write_line(f, f"Estimated Indexing Time: {custom_indexing_time:.4f} seconds")
            write_line(f, f"Estimated Retrieval Time: {custom_retrieval_time:.4f} seconds")

        except Exception as e:
            write_line(f, f"\nError: {e}")


        #2nd model
        try:

            start_custom_indexing2 = time.time()
            analyshEurethriou()
            custom_indexing_time = time.time() - start_custom_indexing2 + time1

            start_custom_retrieval = time.time()
            analyshErwthsewn()
            findDocumentRanks()
            custom_retrieval_time = time.time() - start_custom_retrieval

            with open(f'textFiles/sortedRelevant{2}.json', 'r') as jf:
                custom_ranks = json.load(jf)

            custom_metrics = evaluate_custom_model(custom_ranks, relevant_docs)

            write_line(f, f"\nMetrics for Best fully weighted system tfc-nfx Model:")
            for metric, value in custom_metrics.items():
                write_line(f, f"  {metric}: {value:.4f}")
            write_line(f, f"Estimated Indexing Time: {custom_indexing_time:.4f} seconds")
            write_line(f, f"Estimated Retrieval Time: {custom_retrieval_time:.4f} seconds")

        except Exception as e:
            write_line(f, f"\nError: {e}")

    