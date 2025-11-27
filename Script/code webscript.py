import requests
from bs4 import BeautifulSoup

def get_comments(base_url, start_page, end_page):
    for page in range(start_page, end_page + 1):
        # Construire l'URL de la page actuelle
        url = f'{base_url}?page={page}'

        # Envoyer une requête HTTP à l'URL
        response = requests.get(url)

        # Analyser le contenu de la page avec BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Trouver les éléments contenant les commentaires
        comments = soup.find_all('div', class_='content-txt review-card-content')

        # Extraire et afficher le texte de chaque commentaire
        for comment in comments:
            print(comment.get_text())

# Liste des URLs de base des sections des critiques pour différents films
films_urls = [
    'https://www.allocine.fr/film/fichefilm-267218/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-10700/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-251105/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-311895/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-316875/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-270235/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-294352/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-287126/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-236415/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-173087/critiques/spectateurs/',
    'https://www.allocine.fr/film/fichefilm-260600/critiques/spectateurs/',
    'https://www.allocine.fr/series/ficheserie-25364/critiques/',
    'https://www.allocine.fr/series/ficheserie-23331/critiques',
    'https://www.allocine.fr/series/ficheserie-10001/critiques/',
    'https://www.allocine.fr/series/ficheserie-7157/critiques/',
    'https://www.allocine.fr/series/ficheserie-11303/critiques/',
    'https://www.allocine.fr/series/ficheserie-26316/critiques/',
    'https://www.allocine.fr/series/ficheserie-322/critiques/',
    'https://www.allocine.fr/series/ficheserie-322/critiques/',
    'https://www.allocine.fr/series/ficheserie-290/critiques/',
    # Ajoutez d'autres URLs ici
]

# Paramètres pour le nombre de pages à parcourir pour chaque film
start_page = 1
end_page = 2  # Ajustez selon le besoin

# Parcourir chaque URL de film et récupérer les commentaires
for film_url in films_urls:
    get_comments(film_url, start_page, end_page)
