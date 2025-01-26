from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(train, train_submission):
    """
    Entraîner un modèle Random Forest
    """
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=5,
                                   random_state=1)
    model.fit(train, train_submission)
    return model


def save_model(model, filepath: str):
    """
    Sauvegarder le modèle
    """
    joblib.dump(model, filepath)


if __name__ == "__main__":
    """
    Test simple des fonctions
    """
    from data_preprocessing import tel_data, preprocess_data

    # Charger les données
    train_path = "train.csv"
    test_path = "test.csv"
    train_data, test_data = tel_data(train_path, test_path)
    features = ["Pclass", "Sex",
                "SibSp", "Parch"]

    # Première fonction
    X_train, X_test, y_train = preprocess_data(train_data,
                                               test_data,
                                               features)

    # Deuxième fonction
    model = train_model(X_train, y_train)

    # Sauvegarder le modèle
    save_model(model, "random_forest_model.pkl")
    print("Modèle sauvegardé")
