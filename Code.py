from googleapiclient.discovery import build
import pymongo
import requests
import json
import string
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
import pandas as pd
from googleapiclient.discovery import build
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


#1-1



# Clé d'API YouTube
api_key = 'AIzaSyCt1Xf6QiTYtUPH7I1_YYpiKE4TydZlt_g'

# Création de l'objet de service YouTube
youtube = build('youtube', 'v3', developerKey=api_key)

# Paramètres de recherche
num_videos = 10 # Nombre de vidéos à sélectionner
categories = ['Music', 'Sports', 'Gaming', 'Travel'] # Catégories souhaitées

# Récupération des vidéos populaires dans les catégories spécifiées
videos = []
for category in categories:
    request = youtube.videos().list(
        part='snippet',
        chart='mostPopular',
        regionCode='MA',
        videoCategoryId=category,
        maxResults=num_videos
        )
    response = request.execute()
    videos.extend(response['items'])

# Sélection aléatoire de vidéos
selected_videos = random.sample(videos, num_videos)

# Affichage des informations des vidéos sélectionnées
for video in selected_videos:
    video_id = video['id']
    video_title = video['snippet']['title']
    video_category = video['snippet']['categoryId']
print(f"Video ID: {video_id}")
print(f"Title: {video_title}")
print(f"Category: {video_category}")
print("--------------------")


#1-2

# Fonction pour récupérer les informations des vidéos et leurs commentaires
def import_video_data(video_id):
    # Configuration de la requête API pour récupérer les informations de la vidéo
    video_request = youtube.videos().list(
        part='snippet',
        id=video_id
        )
    # Exécution de la requête API pour récupérer les informations de la vidéo
    video_response = video_request.execute()
    video_data = video_response['items'][0]['snippet']
    # Configuration de la requête API pour récupérer les commentaires de la vidéo
    comments_request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id
        )
    # Exécution de la requête API pour récupérer les commentaires de la vidéo
    comments_response = comments_request.execute()
    comments = []
    for comment in comments_response['items']:
        comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment_text)
    # Stockage des données dans MongoDB
    video_data['comments'] = comments
    collection.insert_one(video_data)


# Liste des identifiants des vidéos à importer
video_ids = ['VIDEO_ID_1', 'VIDEO_ID_2',
'VIDEO_ID_3'] # Remplacez par les identifiants des vidéos que vous souhaitez importer

# Import des données pour chaque vidéo
for video_id in video_ids:
    import_video_data(video_id)


#1-2



# Récupération des données de la collection
data = collection.find()

# Conversion des données en format JSON
json_data = json.dumps(list(data), indent=4) # Indentation pour une meilleure lisibilité

# Écriture des données dans un fichier JSON
with open('youtube_data.json', 'w') as json_file:
    json_file.write(json_data)

#2-



# Téléchargement des ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Fonction de nettoyage des commentaires
def clean_text(text):
    # Suppression des émojis
    text = demojize(text)
    text = re.sub(r':[a-zA-Z0-9_]+:', ' ', text)
    # Suppression des liens
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Suppression de la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Conversion en minuscules
    text = text.lower()
    # Tokenization des mots
    words = word_tokenize(text)
    # Suppression des stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Reconstitution du texte
    cleaned_text = ' '.join(words)
    return cleaned_text


# Liste des commentaires
comments = [
"This is a sample comment with 😃emojis, links: https://example.com and some punctuation! #NLP",
"Another comment with emojis 😊 and some repetitive words words words...",
"A third comment with links: http://example.com, https://example.org"
]

# Nettoyage des commentaires
cleaned_comments = [clean_text(comment) for comment in comments]

# Filtrage des commentaires répétés
filtered_comments = list(set(cleaned_comments))

# Transformation en TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(filtered_comments)
feature_names = vectorizer.get_feature_names()

# Affichage des résultats
for i, comment in enumerate(filtered_comments):
    print(f"Comment {i + 1}:")
    print(comment)
    print("TF-IDF:")
    for j, feature_index in enumerate(tfidf_matrix[i].indices):
        feature_name = feature_names[feature_index]
        tfidf_score = tfidf_matrix[i, feature_index]
        print(f"{feature_name}: {tfidf_score}")
        print("-------------------")


#3-1

# Chargement des données
data = pd.read_csv('commentaires.csv') # Assurez-vous d'avoir le fichier CSV correspondant à vos commentaires

# Annotation des données
annotations = {}

def annotate_comment(comment_id, polarity):
    annotations[comment_id] = polarity
    def next_comment():
        # Affiche le commentaire suivant à annoter
        comment_id = comment_ids.pop(0)
        comment = data.loc[comment_id, 'comment']
        comment_label.config(text=comment)
        yes_button.config(state=tk.NORMAL)
        no_button.config(state=tk.NORMAL)

def save_annotations():
    # Enregistre les annotations dans un fichier CSV
    annotated_data = pd.DataFrame(annotations.items(), columns=['comment_id', 'polarity'])
    annotated_data.to_csv('annotations.csv', index=False)
def annotate_yes():
        annotate_comment(comment_ids[0], 'positive')
        next_comment()

def annotate_no():
    annotate_comment(comment_ids[0], 'negative')
    next_comment()

# Création de l'interface utilisateur
window = tk.Tk()

comment_label = tk.Label(window, text='')
comment_label.pack()

yes_button = tk.Button(window, text='Positive', command=annotate_yes, state=tk.DISABLED)
yes_button.pack()

no_button = tk.Button(window, text='Negative', command=annotate_no, state=tk.DISABLED)
no_button.pack()

comment_ids = list(data.index)
next_comment()

window.mainloop()

# Enregistrement des annotations
save_annotations()

#3-2



# Exemple de données
comments = [
"Ce film est vraiment excellent !",
"Je n'ai pas du tout aimé cette chanson.",
"Les vacances étaient superbes. Je me suis bien amusé !",
"J'ai détesté la finale du match. L'arbitrage était horrible."
]

# Nettoyage des données
cleaned_comments = []
for comment in comments:
    # Suppression des caractères indésirables
    clean_comment = re.sub(r'[^\w\s]', '', comment)
    cleaned_comments.append(clean_comment)

# Tokenisation
tokenized_comments = [word_tokenize(comment) for comment in cleaned_comments]

# Normalisation
normalized_comments = []
for tokens in tokenized_comments:
    # Conversion en minuscules
    normalized_tokens = [token.lower() for token in tokens]
    normalized_comments.append(normalized_tokens)

# Filtrage des commentaires répétés
unique_comments = list(set([' '.join(tokens) for tokens in normalized_comments]))

# Transformation en vecteurs TF-IDF
vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'))
X = vectorizer.fit_transform(unique_comments)

#3-3


from sklearn.model_selection import train_test_split

# X : données d'entrée (commentaires préparés en vecteurs)
# y : étiquettes de polarité correspondantes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Prédictions du modèle
y_pred = model.predict(X_test)

# Calcul des métriques d'évaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Définition du nombre de plis (folds)
num_folds = 5

# Initialisation du validateur croisé
kf = KFold(n_splits=num_folds, random_state=42, shuffle=True)

# Boucle sur les plis
fold_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Entraînement et évaluation du modèle
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
fold_score = accuracy_score(y_test, y_pred)
fold_scores.append(fold_score)

# Calcul de la moyenne des scores des plis
mean_score = sum(fold_scores) / num_folds

print("Mean Accuracy:", mean_score)

#3-4

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Préparation des données (X_train, X_test, y_train, y_test)

# Création d'une instance du classifieur
classifier = LogisticRegression()

# Entraînement du classifieur
classifier.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = classifier.predict(X_test)

# Évaluation des performances du classifieur
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#3-5

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print("Precision:", precision)
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print("Recall:", recall)
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)
from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
