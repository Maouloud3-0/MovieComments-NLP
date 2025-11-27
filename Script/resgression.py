
import pandas as pd
import re
import spacy
from textblob import TextBlob
import emoji
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("C:/Users/gaous/Desktop/BUT 3/S6/R6 06 machine learning texte/commentaires_films_series.csv")

# Charger le modèle de langue française
nlp = spacy.load('fr_core_news_sm')

# Fonction pour compter les émoticônes
def compter_emoticones(texte):
    return emoji.emoji_count(texte)


# Fonction pour compter les mots
def compter_mots(texte):
    return len(texte.split())

# Fonction pour compter les verbes, entités nommées et adjectifs
def compter_pos_entites(texte):
    doc = nlp(texte)
    nb_verbes = len([token for token in doc if token.pos_ == "VERB"])
    nb_entites = len(doc.ents)
    nb_adjectifs = len([token for token in doc if token.pos_ == "ADJ"])
    return nb_verbes, nb_entites, nb_adjectifs

import spacy
from textblob import TextBlob

# Chargez le modèle de langue français de spaCy
nlp = spacy.load('fr_core_news_sm')

def temps_dominant(texte):
    doc = nlp(texte)
    temps_verbes = {'present': 0, 'past': 0, 'future': 0}
    
    for token in doc:
        if token.pos_ == 'VERB' or token.pos_ == 'AUX':
            if token.morph.get("Tense") == ['Pres']:
                temps_verbes['present'] += 1
            elif token.morph.get("Tense") == ['Past']:
                temps_verbes['past'] += 1
            elif token.morph.get("Mood") == ['Cnd'] or token.text.lower() in ['sera', 'auront', 'iront']:
                temps_verbes['future'] += 1
    
    # Trouvez le temps le plus utilisé et renvoyez-le
    return max(temps_verbes, key=temps_verbes.get), temps_verbes




from textblob import TextBlob

def analyser_sentiment_verbes(texte):
    doc = nlp(texte)
    sentiments = {'positifs': 0, 'negatifs': 0}
    
    for token in doc:
        if token.pos_ == 'VERB':
            verdict = TextBlob(str(token))
            if verdict.sentiment.polarity > 0:
                sentiments['positifs'] += 1
            elif verdict.sentiment.polarity < 0:
                sentiments['negatifs'] += 1
    
    return sentiments

# Assurez-vous que 'nlp' est bien défini et initialisé, par exemple:
# nlp = spacy.load('fr_core_news_sm') ou le modèle de votre choix



# Application des fonctions
df['Nb_Mots'] = df['Commentaire'].apply(compter_mots)
df['Nb_Emoticones'] = df['Commentaire'].apply(compter_emoticones)

# Compter les verbes, entités nommées et adjectifs et les ajouter au DataFrame
df[['Nb_Verbes', 'Nb_Entites_Nommees', 'Nb_Adjectifs']] = df['Commentaire'].apply(
    lambda x: pd.Series(compter_pos_entites(x))
)

# Appliquer les fonctions sur le DataFrame
# Appliquer la fonction modifiée pour obtenir le temps dominant et le décompte des verbes
df['Temps_Dominant'], df['Compte_Temps'] = zip(*df['Commentaire'].apply(temps_dominant))

# Créer des colonnes individuelles pour chaque temps à partir du dictionnaire retourné
df['Nb_Verbes_Present'] = df['Compte_Temps'].apply(lambda x: x['present'])
df['Nb_Verbes_Past'] = df['Compte_Temps'].apply(lambda x: x['past'])
df['Nb_Verbes_Future'] = df['Compte_Temps'].apply(lambda x: x['future'])

# Vous pouvez ensuite supprimer la colonne 'Compte_Temps' si elle n'est pas nécessaire
df.drop(columns=['Compte_Temps'], inplace=True)

# Appliquez la fonction à votre DataFrame
df['Sentiment_Verbes'] = df['Commentaire'].apply(analyser_sentiment_verbes)


# Afficher le DataFrame pour vérifier
df


# Enregistrer le DataFrame en fichier CSV
df.to_csv('C:/Users/gaous/Desktop/BUT 3/S6/R6 06 machine learning texte/commentaire_avec_critères.csv', index=False)

df.columns

#
# Créer la colonne cible en fonction du score
df['Cible'] = (df['Note'] > 3.5).astype(int)
# Initialiser le LabelEncoder
le = LabelEncoder()

# Supposons que 'Colonne_Catégorielle' est le nom de votre colonne catégorielle
df['Temps_Dominant'] = le.fit_transform(df['Temps_Dominant'])

# Sélectionner les caractéristiques pour la régression logistique

X = df[['Nb_Mots', 'Nb_Emoticones', 'Nb_Verbes','Nb_Entites_Nommees', 'Nb_Adjectifs', 
        'Temps_Dominant','Nb_Emoticones', 'Nb_Verbes_Present', 'Nb_Verbes_Past', 'Nb_Verbes_Future']]
y = df['Cible']

# Supposons que 'X' sont les caractéristiques et 'y' est la variable cible.
# Diviser les données, en utilisant à la fois 'train_size' et 'test_size', avec stratification et mélange
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.17,     # 20% des données comme ensemble de test
    train_size=0.83,    # 80% des données comme ensemble d'entraînement
    random_state=42,   # Pour la reproductibilité
    shuffle=True,      # Mélanger les données avant de diviser
    stratify=y         # Pour conserver la proportion des classes dans les splits
)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédire les valeurs pour l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"L'exactitude du modèle est : {accuracy}")
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, log_loss

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Matrice de confusion :\n{conf_matrix}")

# Rapport de classification
class_report = classification_report(y_test, y_pred)
print(f"Rapport de classification :\n{class_report}")

# Précision, Rappel, Score F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Précision : {precision}")
print(f"Rappel : {recall}")
print(f"Score F1 : {f1}")

# ROC AUC
y_pred_proba = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC : {roc_auc}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Prédictions de probabilité pour la classe positive
y_pred_proba = model.predict_proba(X_test)[:,1]

# Calcul des taux de vrais positifs et faux positifs
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calcul de l'aire sous la courbe (AUC)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Log Loss
logloss = log_loss(y_test, y_pred_proba)
print(f"Log Loss : {logloss}")
