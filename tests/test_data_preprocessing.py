import pandas as pd
from src.data_preprocessing import tel_data, preprocess_data


def test_tel_data(tmp_path):
    """
    Test pour la fonction tel_data.
    """
    # Créer des fichiers CSV temporaires
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    # Contenu des fichiers
    train_csv.write_text(
        "Pclass,Sex,SibSp,Parch,Survived\n1,male,0,0,1\n2,female,1,0,0"
    )
    test_csv.write_text("Pclass,Sex,SibSp,Parch\n3,male,0,0\n1,female,1,2")

    # Charger les données
    train_data, test_data = tel_data(str(train_csv), str(test_csv))

    # Vérifications avec réduction des lignes
    assert (
        isinstance(train_data, pd.DataFrame)
        and "train_data doit être un DataFrame"
    )
    assert (
        isinstance(test_data, pd.DataFrame)
        and "test_data doit être un DataFrame"
    )
    assert (
        train_data.shape == (2, 5)
        and "Les dimensions de train_data ne correspondent pas"
    )
    assert (
        test_data.shape == (2, 4) and "Les dimensions de test_data ne correspondent pas"
    )


def test_preprocess_data():
    """
    Test pour la fonction preprocess_data.
    """
    # Créer des DataFrames factices
    train_data = pd.DataFrame(
        {
            "Pclass": [1, 2],
            "Sex": ["male", "female"],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Survived": [1, 0],
        }
    )
    test_data = pd.DataFrame(
        {
            "Pclass": [3, 1],
            "Sex": ["male", "female"],
            "SibSp": [0, 1],
            "Parch": [0, 2]}
    )
    features = ["Pclass", "Sex", "SibSp", "Parch"]

    # Prétraiter les données
    train, test, train_submission = preprocess_data(train_data, test_data, features)

    # Vérifications avec réduction des lignes
    assert (
        isinstance(train, pd.DataFrame)
        and "train doit être un DataFrame"
    )
    assert (
        isinstance(test, pd.DataFrame)
        and "test doit être un DataFrame"
    )
    assert (
        isinstance(train_submission, pd.Series)
        and "train_submission doit être un Series"
    )
    assert (
        train.shape[1] == test.shape[1]
        and "Le nombre de colonnes entre train et test doit être le même"
    )
    assert (
        list(train_submission) == [1, 0]
        and "Les valeurs de train_submission ne sont pas correctes"
    )
