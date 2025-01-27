import pytest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.model_evaluation import tel_model, gener_submission


def test_tel_model(tmp_path):
    """
    Test pour la fonction tel_model.
    """
    # Créer un modèle factice
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit([[1, 0], [0, 1]], [1, 0])

    # Sauvegarder le modèle dans un fichier temporaire
    model_path = tmp_path / "random_forest_model.pkl"
    joblib.dump(model, model_path)

    # Charger le modèle avec la fonction tel_model
    loaded_model = tel_model(str(model_path))

    # Vérifications
    assert isinstance(
        loaded_model, RandomForestClassifier
    ), "Le modèle chargé doit être une instance de RandomForestClassifier"
    assert loaded_model.n_estimators == 10, (
        "Le modèle chargé doit avoir le bon paramètre n_estimators"
    )


def test_gener_submission(tmp_path):
    """
    Test pour la fonction gener_submission.
    """
    # Créer un modèle factice
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Données factices avec noms de colonnes
    train = pd.DataFrame({"feature1": [1, 0], "feature2": [0, 1]})
    train_submission = [1, 0]
    model.fit(train, train_submission)

    # Données de test avec les mêmes noms de colonnes
    test = pd.DataFrame({"feature1": [1, 0], "feature2": [0, 1]})
    test_data = pd.DataFrame({"PassengerId": [1, 2]})
    predictions = model.predict(test)

    # Chemin du fichier de sortie
    output_path = tmp_path / "submission.csv"

    # Générer le fichier de soumission
    gener_submission(model, test, test_data, str(output_path))

    # Charger le fichier généré
    output = pd.read_csv(output_path)

    # Vérifications
    assert "PassengerId" in output.columns, (
        "Le fichier de sortie doit contenir une colonne PassengerId"
    )
    assert "Survived" in output.columns, (
        "Le fichier de sortie doit contenir une colonne Survived"
    )
    assert list(output["PassengerId"]) == [1, 2], (
        "Les PassengerId doivent correspondre aux données test"
    )
    assert list(output["Survived"]) == list(predictions), (
        "Les prédictions doivent correspondre aux résultats du modèle"
    )
