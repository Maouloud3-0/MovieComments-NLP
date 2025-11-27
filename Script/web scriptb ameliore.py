import requests
from bs4 import BeautifulSoup
import random
import csv

def get_comments(base_url, min_rating, max_rating):
    page = 1
    comments_low_rating = []
    comments_high_rating = []

    while True:
        url = f'{base_url}?page={page}'
        response = requests.get(url)

        if not response or response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, 'html.parser')

        # Vérifier s'il y a une page suivante, si non, arrêter la boucle
        next_button = soup.find('span', class_='button-right')
        if not next_button or 'button-disabled' in next_button.get('class', []):
            break

        # Trouver les éléments contenant les notes et les commentaires
        comment_blocks = soup.find_all('div', class_='review-card-review-holder')
        for block in comment_blocks:
            note_span = block.find('span', class_='stareval-note')
            if note_span:
                note_text = note_span.get_text().strip()
                note = float(note_text.replace(',', '.'))
                comment_text = block.find('div', class_='content-txt review-card-content').get_text(strip=True)
                
                # Classer le commentaire selon la note
                if note < min_rating:
                    comments_low_rating.append((note, comment_text))
                elif note > max_rating:
                    comments_high_rating.append((note, comment_text))

        page += 1

    # Déterminer le nombre minimum de commentaires dans les deux listes pour égaliser
    min_comments = min(len(comments_low_rating), len(comments_high_rating))
    selected_low_rating = random.sample(comments_low_rating, min_comments) if comments_low_rating else []
    selected_high_rating = random.sample(comments_high_rating, min_comments) if comments_high_rating else []

    return selected_low_rating, selected_high_rating

def save_comments_to_csv(comments, csv_file_path):
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for comment in comments:
            writer.writerow(comment)

# Paramètres pour les critères de notation
min_rating = 2.4
max_rating = 3.5

# Liste des URLs de base des sections des critiques pour différents films et séries
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
    'https://www.allocine.fr/series/ficheserie-22940/critiques/',
    'https://www.allocine.fr/series/ficheserie-290/critiques/',
    # Ajoutez d'autres URLs ici
]

# Nom du fichier CSV pour enregistrer les données
csv_file_path = 'commentaires_films_series.csv'

# En-tête du fichier CSV
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['URL', 'Note', 'Commentaire'])

# Parcourir chaque URL de film et récupérer les commentaires
for film_url in films_urls:
    comments_low, comments_high = get_comments(film_url, min_rating, max_rating)

    # Enregistrer les commentaires dans le fichier CSV
    all_comments = [(film_url, note, comment) for note, comment in comments_low + comments_high]
    save_comments_to_csv(all_comments, csv_file_path)

    print(f"URL: {film_url}")
    print(f"Nombre de commentaires à faible note récupérés: {len(comments_low)}")
    print(f"Nombre de commentaires à haute note récupérés: {len(comments_high)}")
    

