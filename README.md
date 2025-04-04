# Détection et Classification des Pièces d'Échecs avec Mask R-CNN

## Autrices
- HACHMI Ilham - M1 MASERATI
- TRAORE Assa - M1 MASERATI
- EL HACHMI Séhame - M1 MASERATI


## Objectif du Projet
L'objectif de ce projet est de créer un modèle capable de détecter les pièces d'échecs et de les classer selon leurs types (roi, reine, tour, cavalier, fou, pion).

## Technologies Utilisées
- **Python** (version 3.8.6)
- **TensorFlow** (version 1.15)
- **Keras** (version 2.3.1) avec `tensorflow.compat.v1`
- **Mask R-CNN**
- **Skimage** 
- **Scikit-learn**

## Hyper Paramètres
Les principaux hyperparamètres utilisés dans l'entraînement sont les suivants :

- **BACKBONE** : `resnet101`
- **IMAGES_PER_GPU** : `1`
- **NUM_CLASSES** : `7` 
- **STEPS_PER_EPOCH** : `250`
- **VALIDATION_STEPS** : `50`
- **LEARNING_RATE** : `0.001`
- **DETECTION_MIN_CONFIDENCE** : `0.9`
- **USE_MINI_MASK** : `False`
- **IMAGE_MIN_DIM** : `800`
- **IMAGE_MAX_DIM** : `1024`
- **RPN_ANCHOR_SCALES** : `(32, 64, 128, 256, 512)`
- **RPN_ANCHOR_RATIOS** : `[0.5, 1, 2]`
- **TRAIN_ROIS_PER_IMAGE** : `200`
- **BATCH_SIZE** : `1`


## Données
Le projet utilise un ensemble de données de pièces d'échecs comprenant :
- **Nombre total d'images** : `581`
- **Images d'entrainement** : `461`
- **Images de validation** : `120`

### Structure des Données
Les images et les annotations sont organisées de la manière suivante :
- `train/`: Contient les images et annotations au format JSON pour l'entraînement.
- `val/`: Contient les images et annotations au format JSON pour la validation.

## Rapport
Vous trouverez le code complet attaché au projet au format .ipynb. 
Dû à l'échec d'importation du dossier logs et des données sur GitHub du fait de leur taille (même après compresiion), nous avons choisi de créer un Google Drive où vous retrouverez le logs, les données et les images utilisées dans le reste de ce rapport (matrice de confusion, visualisations et graphique de pertes). Vous pouvez y accéder par le biais de ce lien :

[Accéder aux fichiers Google Drive](https://drive.google.com/drive/folders/1kOq5Tw-TcAt8bjzzpom8F91n0UQp63xT)

### Entrainement

Nous avons lancé notre entrainement avec 11 epochs de 250 pas chacune. Vous retrouverez le dossier logs\ dans le drive accessible par le lien donné plus haut. Nous n'avons gardé dans le dossier logs que les résultats du dernier entrainement, les entrainements d'essais précédemment réalisés ayant été exclus.

Notre entrainement nous affiche ces valeurs de pertes à la fin de la dernière epoch : 

<img width="632" alt="Dernière ligne de l'entrainement" src="https://github.com/user-attachments/assets/ad9178d7-e275-469a-80af-950292b68625">

Nous reviendrons sur les valeurs des pertes tout au long de l'exécution des différentes epochs ultérieurement.

### Visualisation avec données de validation

Après l'entrainement, nous avons lancé des prédictions sur nos données de validation. Avec un DETECTION_MIN_CONFIDENCE initialement paramétré à 0.9, les résultats n'étaient que très peu concluants. Après avoir testé plusieurs valeurs, nous avons retenu un DETECTION_MIN_CONFIDENCE à 0,6, qui nous renvoyait des résultats plus corrects sans être majoritairement erronés. Vous trouverez ci dessous la visualisation de la prédiction sur plusieurs pièces tirées au hasard dans nos données de validation : 

<img width="217" alt="detect reine" src="https://github.com/user-attachments/assets/f99bf906-13da-46e6-86eb-baa77d25db6d">
<img width="228" alt="detect fou" src="https://github.com/user-attachments/assets/8ce2ccdb-d282-4198-8ef1-5cad5fe9f240">
<img width="194" alt="detect cavalier" src="https://github.com/user-attachments/assets/e82c76a4-48fe-48f5-a0d9-28c891fd6ff7">
<img width="151" alt="detec pion" src="https://github.com/user-attachments/assets/6b9eb1ec-0ac1-4a29-a974-4702edc2a92d">
<img width="209" alt="detect roi" src="https://github.com/user-attachments/assets/2008a516-ec31-4039-9cca-9a76c80fc31b">
<img width="205" alt="detect roi 2" src="https://github.com/user-attachments/assets/a2396ed5-03b1-46bd-8e54-038d226b40b5">

Bien que la prédiction soit parfois éronnée ou inexistante à ce niveau de DETECTION_MIN_CONFIDENCE pour certaines images, la prédiction reste dans l'ensemble correcte.

### Résultats

#### Graphique de pertes

Nous tirons de notre entrainement les valeurs des pertes (loss) de chaque epoch exécutée, grâce auxquelles nous construisons le graphique de perte (généré sur R) :

| Epoch | Train Loss | Validation Loss |
|-------|------------|-----------------|
| 1     | 1.1859     | 0.4035          |
| 2     | 1.0839     | 1.1851          |
| 3     | 0.5812     | 1.0321          |
| 4     | 0.5233     | 0.6632          |
| 5     | 0.4827     | 0.4612          |
| 6     | 0.4125     | 0.3025          |
| 7     | 0.3912     | 0.4244          |
| 8     | 0.3572     | 0.1724          |
| 9     | 0.3524     | 0.3817          |
| 10    | 0.3388     | 0.4145          |
| 11    | 0.3023     | 0.1515          |
<img width="502" alt="Courbe loss" src="https://github.com/user-attachments/assets/1d6bd216-4c17-4f49-b494-bf6a9b1cc638">

On remarque une convergence stable des valeurs de nos pertes au fur et à mesure de l'exécution des époques, qui pourrait être améliorée si le nombre d'epochs avait été plus important.

#### Matrice de confusion

Nous tirons de notre modèle la matrice confusion suivante :

<img width="380" alt="Matrice de confusion" src="https://github.com/user-attachments/assets/d769f78e-018a-4334-b448-588ecbb496fd">

On remarque que le modèle ne reconnaît pas les différentes pièces avec la même précision : le cavalier est la pièce la mieux reconnue et n'est confondu avec aucune autre pièce. Cela peut s'expliquer par la forme spécifique du cavalier, qui est relativement singulière et très reconnaissable. Le fou est la pièce la moins reconnaissable par notre modèle, souvent confondu avec le fond (background) ou les pions. Le fait que des pièces comme le pion ou le roi soient également confondues avec d'autres pièces peut s'expliquer par le fait que certaines classes de pièces partagent des caractéristiques similaires en termes de design.


### Conclusion et possibles améliorations

Nous pouvons conclure que notre modèle semble bien fonctionner pour certaines classes de pièces, mais fonctionne de manière médiocre pour d'autres. Plusieurs améliorations sont envisageables pour l'optimiser : entre autres, recueillir davantage de données d'entraînement, en particulier pour les classes où les confusions sont les plus fréquentes, et ajuster les hyperparamètres tels que le nombre d'epochs ou le nombre de pas. La réduction de DETECTION_MIN_CONFIDENCE lors de l'entraînement aurait également pu être envisagée, une valeur trop élevée ayant peut-être poussé notre modèle à ignorer certaines sélections.


## Difficultés Rencontrées
| Problème | Description | Solution |
|----------|-------------|----------|
| **Version de Python** | Ajustement de la version de Python pour assurer la compatibilité avec le MRCNN utilisé. | Création d'un environnement virtuel via l'invite de commande et utilisation de la version 3.8.6 de Python. |
| **Incompatibilités Keras/TensorFlow** | Problèmes récurrents d'import entre les versions Keras et TensorFlow. | Utilisation de `tensorflow.compat.v1` et installation des versions Keras 2.3.1 et Tensorflow 1.15 |
| **Incompatibilité TensorFlow/Mask RCNN** | La version de Mask R-CNN n’était pas compatible avec la version de TensorFlow utilisée. | Utilisation de TensorFlow 1.15 |
| **Création des fichiers d'annotation JSON** | Noms de fichiers d'images se répétant entre les différentes classes, entraînant des conflits dans les annotations. | Modification des chemins et noms de fichiers pour les rendre uniques dans le fichier JSON généré. |
| **Crash du noyau Jupyter** | Le noyau Jupyter a perdu la connexion pendant l'entraînement. | Reprise de l'entraînement en chargeant les poids sauvegardés de la dernière sauvegarde enregistrée dans le dossier logs/. |
| **Incompatibilité Google Colab avec la version requise de TensorFlow** | Google Colab ne supportait pas la version spécifique de TensorFlow utilisée dans le projet. | Réalisation du projet en local sur Jupyter Notebook. |
| **Problèmes de version de TensorFlow** | Erreurs récurrentes liées aux métriques. | Suppression des occurrences de metrics_tensor et utilisation explicite de tf.compat.v1 dans model.py. |
| **Avertissement Skimage** | Avertissement lors de l'utilisation de skimage.draw.polygon au moment du lancement de l'entrainement. | Ajout du paramètre order=0 dans utils.py pour supprimer l’avertissement lors de la création des masques. |
| **Erreur ValueError: zero-size array** | Les annotations JSON contenaient des segmentations vides ou mal formatées, causant une erreur lors de la création des masques. | Modification de la fonction load_mask pour ajouter une vérification des segmentations avant traitement. |
| **Coordonnées dépassant la résolution de l'image** | Annotations avec coordonnées de segmentation dépassant la taille de l'image, causant des erreurs de masques. | Restriction des coordonnées dans les limites de la résolution de l'image pour les segmentation problématiques. |

## Dernier mot sur le projet

Ce premier projet en intelligence artificielle a été très instructif pour nous, nous introduisant à cette discipline par le biais d'un cas pratique de deep learning. Les nombreux défis auxquels nous avons fait face nous ont poussés à nous documenter davantage, ce qui nous a conduit à une meilleure compréhension des processus sous-jacents à la reconnaissance d'images. Ce premier pas dans ce vaste domaine a éveillé notre curiosité et nous a motivés à approfondir nos connaissances pour nos futurs projets.
