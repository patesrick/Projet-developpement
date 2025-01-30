import pandas as pd
import joblib


def tel_model(filepath: str):
    """
    Charger le modèle sauvegardé
    """
    return joblib.load(filepath)


def gener_submission(model, test, test_data, output_path: str):
    """
    Créer un fichier d'évaluation du modèle
    """
    predictions = model.predict(test)
    output = pd.DataFrame(
        {"PassengerId": test_data["PassengerId"], "Survived": predictions}
    )
    output.to_csv(output_path, index=False)
    print("Fichier enregistré")


if __name__ == "__main__":
    """
    Test simple des fonctions
    """
    from data_preprocessing import tel_data, preprocess_data

    # Charger les données
    train_path = "train.csv"
    test_path = "test.csv"
    train_data, test_data = tel_data(train_path, test_path)

    # Prétraiter les données
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    train, test, train_submission = preprocess_data(train_data, test_data, features)

    # Charger le modèle
    model_path = "random_forest_model.pkl"
    model = tel_model(model_path)

    # Générer le fichier de soumission
    gener_submission(model, test, test_data, "submission.csv")
