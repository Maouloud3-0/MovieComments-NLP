import spacy
import Levenshtein as lev
# chargement du modèle pré-entrainé
nlp = spacy.load('fr_core_news_sm')
# pré-traitement du texte
doc = nlp("This is the first sentence for word2vec with python. This is the second sentence. Yet another sentence, each sentence being composed of words.")

# recuperation du mot du texte a) Afficher les embeddings des mots sentence et word.
mots = doc[14]
# affichage du ième mot du texte
print(mots)
# recuperation du mot du texte a) Afficher les embeddings des mots sentence et word.
motw = doc[-1]
# affichage du ième mot du texte
print(motw)

#1| a) Afficher les embeddings des mots sentence et word.
# affichage de l’embedding du ième mot
print(mots.vector)
# affichage de l’embedding du ième mot
print(motw.vector)

# b) Calculer la similarité entre ces deux mots.
print("la similarité est :",mots.similarity(motw))
# le distance 
print(lev.distance("sentence","words"))
print(lev.ratio("sentence","words"))

