# **Titanic Survival Prediction - Projet d'Ingénierie Logicielle**

## **Contexte et Objectif du Projet**

Ce projet vise à transformer un notebook Python existant, tiré du célèbre projet **Titanic Survival Prediction** sur Kaggle, en un ensemble de scripts Python modulaires, bien structurés et réutilisables.
Le projet **Titanic Survival Prediction** se présente en plusieurs étapes fondamentales, notamment :
- Le nettoyage, la préparation et l'analyse des données.
- La réalisation d'une analyse exploratoire (EDA).
- Le développement d'un modèle d'apprentissage automatique (ML) capable de prédire les probabilités de survie des passagers du Titanic.

L'objectif principal est de mettre en pratique les **bonnes pratiques d'ingénierie logicielle** enseignées en cours, en appliquant notamment :
- La conformité aux conventions **PEP 8** pour un code propre et lisible.
- La mise en place de **tests unitaires** pour garantir la robustesse du code.
- L'utilisation de **Git/GitHub** pour collaborer efficacement en équipe.
- La configuration d’un pipeline **CI/CD** avec **GitHub Actions** pour automatiser les tests, le linting et, éventuellement, le déploiement.



---



## **Organisation des fichiers et dossiers du projet**

Comme dit précedemment, ce projet vise à prédire les chances de survie des passagers du Titanic en utilisant des modèles de machine learning. Voici la structure de l'organisation des fichiers et dossiers de ce travail :

```plaintext
Projet-developpement/
│
├── docs/                            # Documentation principale
│   ├── README.md                    # Documentation du projet
│   ├── rapport.pdf                  # Rapport du projet
|
├── src/                             # Scripts Python modulaires & Données Brutes
│   ├── __pycache__/                 # Fichiers compilés Python
│   ├── __init__.py                  # Indique que src est un package Python
│   ├── data_preprocessing.py        # Prétraitement des données
│   ├── model_evaluation.py          # Évaluation des performances
│   ├── model_training.py            # Entraînement du modèle
│   ├── submission.csv               # Export du modèle évalué
│   ├── train.csv                    # Données d'entraînement
│   ├── test.csv                     # Données de test
│
├── tests/                           # Tests unitaires avec pytest
│   ├── __pycache__/                 # Fichiers compilés Python
│   ├── __init__.py                  # Indique que tests est un package Python
│   ├── test_data_preprocessing.py   # Tests pour le prétraitement des données
│   ├── test_model_evaluation.py     # Tests pour l'évaluation du modèle
│   ├── test_model_training.py       # Tests pour l'entraînement du modèle
│
├── README.md                        # Documentation du projet
├── poetry.lock                      # Fichier de verrouillage des dépendances (Poetry)
├── pyproject.toml                   # Configuration de Poetry
```



---



## **Contribution des membres**

4 membres se sont départagés les tâches du projet. Voici une brève description de ce qu'a fait chaque membre du groupe :

### **Tâches**

- **Baptiste TIVRIER** :
  - Familiarisation avec les données de départ
  - Écriture des différents scripts Python
  - Tests de fonctionnement simple et des règles PEP8
  - Co-Écriture du README
  - Utilisation de GIT et structuration de GITHUB

- **Patrick CHEN** :
  - Familiarisation avec les données de départ
  - Mise en place de la gestion des librairies
  - Écriture des différents tests unitaires
  - Co-Écriture du README
  - Utilisation de GIT et structuration de GITHUB

- **Gaspard LUGAT** :
  - Familiarisation avec les données de départ
  - Mise en place de la pipeline CI/CD
  - Co-Écriture du README
  - Utilisation de GIT et structuration de GITHUB

- **Dan SEBAG** :
  - Familiarisation avec les données de départ
  - Écriture et rédaction du rapport final
  - Co-Écriture du README (sur word)
  - Utilisation de GIT et structuration de GITHUB



---



## **Instructions d'installation des packages**

Ce projet repose sur plusieurs bibliothèques Python essentielles. Voici une description des dépendances utilisées et leur rôle dans le projet :

### **Bibliothèques utilisées**

- **`pandas`** :
  - Librairie incontournable pour la manipulation et l'analyse des données.
  - Utilisée pour charger, nettoyer et transformer les données issues des fichiers CSV (données d'entraînement et de test).

- **`joblib`** :
  - Outil performant pour la sauvegarde et le chargement d'objets Python.
  - Utilisé pour enregistrer les modèles entraînés et les recharger pour l'évaluation ou la prédiction.

- **`scikit-learn`** :
  - Librairie phare pour l'apprentissage automatique en Python.
  - Fournit des algorithmes de machine learning, comme le modèle **Random Forest**, utilisé pour prédire les chances de survie des passagers.

- **`pytest`** :
  - Framework dédié à l’écriture et à l’exécution de tests unitaires.
  - Permet de vérifier la fiabilité du code en testant les fonctions principales du projet.



---



## **Installation des dépendances**

#### **Pré-requis**
Assurez-vous d’avoir installé **Python 3.8** ou une version ultérieure.

#### **Étapes d'installation**
1. Installez **Poetry** si ce n'est pas déjà fait :
   ```bash
   pip install poetry
2. Ajoutez les dépendances principales nécessaires à l'exécution du projet :
   ```bash
   poetry add pandas joblib scikit-learn pytest
3. Ajoutez les dépendances de développement pour le testing et le linting :
   ```bash
   poetry add --dev pytest
4. Vérifiez que toutes les dépendances sont correctement installées :
   ```bash
   poetry install
