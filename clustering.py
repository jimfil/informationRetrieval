from tfidf_tuning import load_docs, find_best_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

docs = load_docs()
documents = list(docs.values())
best_model = find_best_model(docs)



vectorizer = TfidfVectorizer(**best_model['params'])
X_tfidf = vectorizer.fit_transform(documents)

# Επιλογή του βέλτιστου K με βάση το silhouette score
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    labels = KMeans(k, random_state=42, n_init=10).fit_predict(X_tfidf)
    sil_scores.append(silhouette_score(X_tfidf, labels))

K = K_range[sil_scores.index(max(sil_scores))]



kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels_tfidf = kmeans.fit_predict(X_tfidf)

sil_tfidf = silhouette_score(X_tfidf, labels_tfidf)   


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_emb = model.encode(documents)

labels_emb = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(X_emb)
sil_emb = silhouette_score(X_emb, labels_emb)


X_tfidf_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())
X_emb_pca   = PCA(n_components=2).fit_transform(X_emb)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(X_tfidf_pca[:,0], X_tfidf_pca[:,1], c=labels_tfidf)
ax1.set_title("TF–IDF Clustering (PCA)")

ax2.scatter(X_emb_pca[:,0], X_emb_pca[:,1], c=labels_emb)
ax2.set_title("Embeddings Clustering (PCA)")
plt.show()
