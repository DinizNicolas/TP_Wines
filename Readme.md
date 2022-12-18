## Installation 

1 - Installer les dépendances du projet

    pip install -r requirements.txt

2 - Lancer l'application

    uvicorn run:app


## Choix du projet

Preprocessing :
Etant donné que les valeurs des différentes colonnes ont des ordres de grandeur différents, on normalise les données.
On sépare ensuite le dataset en train jeux de données : train, test et validation.

Utilisation de Keras :
- pour la simplicité de tester différentes architectures de modèles
- pour la sauvegarde et chargement de modèle simple

Le fichier data/model_data.json :
- ["scaling_data"] Etant donné que l'on normalise les données pour entrainer le modèle, il faut aussi normaliser les donnée que l'on donne au modèle pour prédire la note d'un vin. Pour cela il faut sauvegarder la moyenne et l'écart type pour chaque colonne.
- ["model_infos"] De plus, pour éviter de charger le modèle et l'évaluer, les données d'évaluation et de configuration sont stockées dans ce fichier

Vin parfait :
Pour prédire le vin parfait, on calcule le coefficient de corrélation entre chaque colonne et la colonne qualité.
Etant donné que l'on cherche la qualité la plus haute, si le coefficient est négatif nous prenons le minimum de cette colonne, sinon le maximum.