# **Titanic Survival Prediction - Projet d'Ingénierie Logicielle**

## **Contexte et Objectif du Projet**

Ce projet vise à transformer un notebook Python existant, tiré du célèbre projet **Titanic Survival Prediction** sur Kaggle, en un ensemble de scripts Python modulaires, bien structurés et réutilisables.
Le projet **Titanic Survival Prediction** se présente en plusieurs étapes, notamment :
- La préparation et l'analyse des données.
- La réalisation d'une analyse exploratoire (EDA).
- Le développement d'un modèle d'apprentissage automatique (ML) capable de prédire les probabilités de survie des passagers du Titanic.

L'objectif principal est de mettre en pratique les **bonnes pratiques d'ingénierie logicielle** enseignées en cours, en appliquant notamment :
- La conformité aux conventions **PEP 8** pour un code propre et lisible.
- La mise en place de **tests unitaires** pour garantir la robustesse du code.
- L'utilisation de **Git/GitHub** pour collaborer efficacement en équipe.
- La configuration d’un pipeline **CI/CD** avec **GitHub Actions** pour automatiser les tests, le linting et, éventuellement, le déploiement.

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

### **1. Installation des dépendances**

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
