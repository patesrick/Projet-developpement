import pandas as pd


def tel_data(train_path: str, test_path: str):
    """
    Charger les données train et test
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(train_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    features: list):
    """
    Préparer les données pour l'entraînement et la prédiction :
    - Mettre les colonnes catégoriques au numérique
    - Prendre les colonnes pertinentes
    """
    train = pd.get_dummies(train_data[features])
    test = pd.get_dummies(test_data[features])
    train_submission = train_data["Survived"]
    return train, test, train_submission


if __name__ == "__main__":
    """
    Test simple des fonctions
    """
    # Charger les données
    train_test = "train.csv"
    test_test = "test.csv"
    features = ["Pclass", "Sex",
                "SibSp", "Parch"]

    # Première fonction
    train_data, test_data = tel_data(train_test, test_test)

    # Deuxième fonction
    train, test, train_submission = preprocess_data(train_data,
                                                    test_data,
                                                    features)

    # Résultats
    print("\nDonnées train prétraitées :")
    print(train.head())

    print("\nDonnées test prétraitées :")
    print(test.head())

    print("\nDonnées submission :")
    print(train_submission.head())
