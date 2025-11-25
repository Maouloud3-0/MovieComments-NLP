# Classification de commentaires de films en NLP

![NLP](https://img.shields.io/badge/NLP-Text_Classification-blue)
![Status](https://img.shields.io/badge/Status-Ternimer-red)
![License](https://img.shields.io/badge/License-MIT-green)

Projet de modélisation statistique et d’IA sur texte appliqué à des **commentaires de films** récupérés par web scraping (type AlloCiné).  
Objectif : construire une **pipeline complète** allant de la collecte des avis jusqu’à la **classification automatique** (sentiment, note, ou catégorie).

---

## 🗂 Organisation du dépôt

```text
.
├── Data/        # Données brutes et nettoyées (CSV, RDS, …)
├── Script/      # Scripts d’analyse, de NLP et de modélisation
├── figures/     # Graphiques et visualisations générées
└── Reports/     # Comptes rendus, rapports (HTML, PDF, Rmd, …)

📊 Données

Source : commentaires de films récupérés par scraping sur un site de critiques.

Contenu typique :

texte du commentaire

titre du film

note / rating

métadonnées (date, pseudo, etc. si disponibles)

Les fichiers nettoyés sont stockés dans Data/processed_*.

🔁 Pipeline d’analyse

Scraping & import

Récupération des pages de critiques

Extraction des champs utiles

Pré-traitement NLP

nettoyage (lowercase, suppression ponctuation, stopwords, etc.)

tokenisation, lemmatisation / stemming

Représentation des textes

sac de mots, TF-IDF

éventuellement embeddings (Word2Vec, fastText, BERT…)

Modélisation

séparation train/test

modèles : régression logistique, SVM, arbres, réseaux de neurones, …

sélection de modèle et réglage d’hyper-paramètres

Évaluation & visualisation

accuracy, F1-score, AUC, etc.

matrices de confusion, courbes ROC

graphiques et tableaux stockés dans figures/ et Reports/.

💻 Prérequis & installation

R ou Python (au choix selon les scripts utilisés)

Packages principaux (exemples) :

R : tidyverse, tidytext, quanteda, caret, rmarkdown

Python : pandas, scikit-learn, matplotlib, seaborn, numpy, beautifulsoup4, requests
