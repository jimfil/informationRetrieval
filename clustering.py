import json
from tfidf_tuning import load_docs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

docs = load_docs()
documents = list(docs.values())
with open("textFiles/bestModel.json", "r") as f: 
    best_model = json.load(f)
best_model["params"]["ngram_range"] = tuple(best_model["params"]["ngram_range"])

vectorizer = TfidfVectorizer(**best_model['params'])
X_tfidf = vectorizer.fit_transform(documents)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_emb = model.encode(documents)

K_range = range(2, 11)

def get_best_k(data, k_range):
    scores = []
    for k in k_range:
        labels = KMeans(k, random_state=42, n_init=10).fit_predict(data)
        scores.append(silhouette_score(data, labels))
    best_k = k_range[scores.index(max(scores))]
    return best_k

k_tfidf_best = get_best_k(X_tfidf, K_range)
k_emb_best = get_best_k(X_emb, K_range)

print(f"Best K for TF-IDF: {k_tfidf_best}")
print(f"Best K for Embeddings: {k_emb_best}")

X_tfidf_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())
X_emb_pca = PCA(n_components=2).fit_transform(X_emb)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

labels1 = KMeans(n_clusters=k_tfidf_best, random_state=42, n_init=10).fit_predict(X_tfidf)
axs[0, 0].scatter(X_tfidf_pca[:,0], X_tfidf_pca[:,1], c=labels1, cmap='viridis')
axs[0, 0].set_title(f"TF-IDF (Best K={k_tfidf_best})")

labels2 = KMeans(n_clusters=k_tfidf_best, random_state=42, n_init=10).fit_predict(X_emb)
axs[0, 1].scatter(X_emb_pca[:,0], X_emb_pca[:,1], c=labels2, cmap='viridis')
axs[0, 1].set_title(f"Embeddings (using TF-IDF K={k_tfidf_best})")

labels3 = KMeans(n_clusters=k_emb_best, random_state=42, n_init=10).fit_predict(X_emb)
axs[1, 0].scatter(X_emb_pca[:,0], X_emb_pca[:,1], c=labels3, cmap='plasma')
axs[1, 0].set_title(f"Embeddings (Best K={k_emb_best})")

labels4 = KMeans(n_clusters=k_emb_best, random_state=42, n_init=10).fit_predict(X_tfidf)
axs[1, 1].scatter(X_tfidf_pca[:,0], X_tfidf_pca[:,1], c=labels4, cmap='plasma')
axs[1, 1].set_title(f"TF-IDF (using Embeddings K={k_emb_best})")

plt.tight_layout()
plt.show()