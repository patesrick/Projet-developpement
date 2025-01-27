import os
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.model_training import train_model, save_model


def test_train_model():
    """
    Test pour vérifier l'entraînement du modèle.
    """
    # Données factices pour l'entraînement
    train = [[1, 0, 1], [2, 1, 0], [3, 0, 1], [4, 1, 0]]
    train_submission = [1, 0, 1, 0]

    # Entraîner le modèle
    model = train_model(train, train_submission)

    # Vérifications
    assert isinstance(model, RandomForestClassifier), (
        "Le modèle doit être une instance de RandomForestClassifier"
    )
    assert hasattr(model, "predict"), (
        "Le modèle entraîné doit avoir une méthode 'predict'"
    )
    assert model.n_estimators == 100, "Le modèle doit utiliser 100 arbres"
    assert model.max_depth == 5, "La profondeur maximale du modèle doit être 5"


def test_save_model(tmp_path):
    """
    Test pour vérifier la sauvegarde du modèle.
    """
    # Données factices pour l'entraînement
    train = [[1, 0, 1], [2, 1, 0], [3, 0, 1], [4, 1, 0]]
    train_submission = [1, 0, 1, 0]

    # Entraîner le modèle
    model = train_model(train, train_submission)

    # Chemin de sauvegarde temporaire
    filepath = tmp_path / "random_forest_model.pkl"

    # Sauvegarder le modèle
    save_model(model, str(filepath))

    # Vérifications
    assert os.path.exists(filepath), "Le fichier de sauvegarde doit exister"
