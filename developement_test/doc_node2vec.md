# 📌 Fiche Récapitulative - Node2Vec

## 🔍 Introduction
Node2Vec permet de générer des **embeddings** (vecteurs numériques) pour représenter les nœuds d'un graphe en utilisant des **marches aléatoires biaisées** et un modèle **Skip-Gram**.

---

## 🚶 Marches Aléatoires
Une marche aléatoire est une séquence de nœuds générée en commençant à un nœud initial et en se déplaçant **aléatoirement** vers les voisins.

### ⚙️ Paramètres principaux
| Paramètre | Description | Effet |
|------------|-------------|-------|
| **p** (*return parameter*) | Probabilité de revenir au nœud précédent | **Valeur élevée** : exploration locale (DFS-like) |
| **q** (*in-out parameter*) | Probabilité de s'éloigner du nœud précédent | **Valeur élevée** : exploration large (BFS-like) |
| **walk_length** | Nombre de nœuds visités par marche aléatoire | **Court** : relations locales, **Long** : capture globale |
| **num_walks** | Nombre de marches aléatoires par nœud | **Faible** : moins de relations, **élevé** : meilleure capture des liens |

📌 **Exemple** :
- Si `walk_length = 30`, chaque marche visite **30 nœuds** avant de s'arrêter.
- Si `num_walks = 200`, chaque nœud est un point de départ pour **200 marches**.

---

## 🔢 Modèle Skip-Gram
Le **Skip-Gram** apprend à **prédire les voisins d'un nœud** dans une marche aléatoire.

1. Pour chaque nœud **central**, le modèle prédit les **nœuds contextuels**.
2. La prédiction se fait dans une **fenêtre** de taille `window`.

📌 **Fonction de perte** :
- Minimise l'erreur de prédiction.
- Ajuste les embeddings pour optimiser la représentation du graphe.

---

## 📤 Résultat : Embeddings
Chaque nœud du graphe est représenté par un vecteur **de dimension fixe**.

✅ **Avantages** :
- Les nœuds **similaires** (proches dans le graphe) ont des **embeddings similaires**.
- Peut être utilisé pour :
  - Classification de nœuds 🏷️
  - Clustering 🔍
  - Recommandation 🤝
  - Prédiction de liens 🔗

---

---

## 🎯 Conclusion
- Node2Vec est **puissant** pour extraire des **représentations vectorielles** de graphes.
- Son **exploration biaisée** permet d’adapter la capture des relations en fonction des besoins (BFS vs DFS).
- Utilisé pour de nombreuses **applications en Machine Learning et Data Science** 🚀.

