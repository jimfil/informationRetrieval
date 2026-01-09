import matplotlib.pyplot as plt

def precision(retrieved_docs, relevant_docs):  # false positives (eggrafa pou anakththikan kai einai lathos)
    """
    Υπολογίζει την Ακρίβεια (Precision).
    Precision = (Αριθμός σχετικών εγγράφων που ανακτήθηκαν) / (Συνολικός αριθμός εγγράφων που ανακτήθηκαν)

    :param retrieved_docs: Λίστα με τα IDs των ανακτημένων εγγράφων.
    :param relevant_docs: Λίστα με τα IDs των σχετικών εγγράφων.
    :return: Η τιμή της ακρίβειας (float).
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    if not retrieved_docs:
        return 0.0
        
    return true_positives / len(retrieved_docs)

def recall(retrieved_docs, relevant_docs): # false negatives (eggrafa pou DEN anakththikan kai einai SWSTA)
    """
    Υπολογίζει την Ανάκληση (Recall).
    Recall = (Αριθμός σχετικών εγγράφων που ανακτήθηκαν) / (Συνολικός αριθμός σχετικών εγγράφων)

    :param retrieved_docs: Λίστα με τα IDs των ανακτημένων εγγράφων.
    :param relevant_docs: Λίστα με τα IDs των σχετικών εγγράφων.
    :return: Η τιμή της ανάκλησης (float).
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    if not relevant_docs:
        return 0.0
        
    return true_positives / len(relevant_docs)

def f1_score(retrieved_docs, relevant_docs):
    p = precision(retrieved_docs, relevant_docs)
    r = recall(retrieved_docs, relevant_docs)
    
    if p + r == 0:
        return 0.0
        
    return 2 * (p * r) / (p + r)

def precision_at_k(retrieved_docs, relevant_docs, k):
    """
    Υπολογίζει την Ακρίβεια@k (Precision@k).
    Precision@k = (Αριθμός σχετικών εγγράφων στα πρώτα k) / k

    :param retrieved_docs: Ταξινομημένη λίστα με τα IDs των ανακτημένων εγγράφων.
    :param relevant_docs: Λίστα με τα IDs των σχετικών εγγράφων.
    :param k: Ο αριθμός των κορυφαίων εγγράφων που θα εξεταστούν.
    :return: Η τιμή της ακρίβειας@k (float).
    """
    if k == 0:
        return 0.0
    
    top_k_docs = retrieved_docs[:k]
    return precision(top_k_docs, relevant_docs)

def recall_at_k(retrieved_docs, relevant_docs, k):
    """
    Υπολογίζει την Ανάκληση@k (Recall@k).
    Recall@k = (Αριθμός σχετικών εγγράφων στα πρώτα k) / (Συνολικός αριθμός σχετικών εγγράφων)  

    :param retrieved_docs: Ταξινομημένη λίστα με τα IDs των ανακτημένων εγγράφων.
    :param relevant_docs: Λίστα με τα IDs των σχετικών εγγράφων.
    :param k: Ο αριθμός των κορυφαίων εγγράφων που θα εξεταστούν.
    :return: Η τιμή της ακρίβειας@k (float).
    """
    if k == 0:
        return 0.0
    
    top_k_docs = retrieved_docs[:k]
    return recall(top_k_docs, relevant_docs)



def plot_precision_recall_curve(retrieved_docs, relevant_docs):
    """
    Υπολογίζει τα σημεία και σχεδιάζει το διάγραμμα Ανάκλησης-Ακρίβειας.

    :param retrieved_docs: Ταξινομημένη λίστα με τα IDs των ανακτημένων εγγράφων.
    :param relevant_docs: Λίστα με τα IDs των σχετικών εγγράφων.
    """
    relevant_set = set(relevant_docs)
    total_relevant = len(relevant_set)
    
    if total_relevant == 0:
        print("Δεν υπάρχουν σχετικά έγγραφα για να δημιουργηθεί το διάγραμμα.")
        return

    recall_points = []
    precision_points = []
    true_positives = 0

    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            true_positives += 1
            current_recall = true_positives / total_relevant
            current_precision = true_positives / (i + 1)
            recall_points.append(current_recall)
            precision_points.append(current_precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_points, precision_points, marker='o', linestyle='-')
    plt.xlabel("Ανάκληση (Recall)")
    plt.ylabel("Ακρίβεια (Precision)")
    plt.title("Διάγραμμα Ακρίβειας-Ανάκλησης (Precision-Recall Curve)")
    plt.grid(True)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.show()