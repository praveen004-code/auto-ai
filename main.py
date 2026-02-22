import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(data):

    # Remove duplicates
    data = data.drop_duplicates()

    # Fill missing values
    data = data.fillna(data.mean(numeric_only=True))

    # Encode categorical columns
    encoder = LabelEncoder()

    for col in data.select_dtypes(include='object').columns:
        data[col] = encoder.fit_transform(data[col])

    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
  from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_models():

    models = {

        "Random Forest": RandomForestClassifier(),

        "Logistic Regression": LogisticRegression(max_iter=1000),

        "XGBoost": XGBClassifier(eval_metric='logloss')

    }

    return models
  import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def optimize_model(X_train, X_test, y_train, y_test):

    def objective(trial):

        n_estimators = trial.suggest_int("n_estimators", 50, 200)

        model = RandomForestClassifier(n_estimators=n_estimators)

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=10)

    best_model = RandomForestClassifier(
        n_estimators=study.best_params["n_estimators"]
    )

    best_model.fit(X_train, y_train)

    return best_model
  import shap

def explain_model(model, X_train):

    explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_train)

    return shap_values
  from preprocessing import preprocess_data
from model_selection import get_models
from optimization import optimize_model
from explainability import explain_model

from sklearn.metrics import accuracy_score


def run_autonomous_ai(data):

    # Step 1: preprocess
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 2: model selection
    models = get_models()

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        score = accuracy_score(y_test, preds)

        if score > best_score:

            best_score = score
            best_model = model
            best_name = name

    # Step 3: optimize best model
    optimized_model = optimize_model(X_train, X_test, y_train, y_test)

    optimized_score = accuracy_score(
        y_test, optimized_model.predict(X_test)
    )

    # Step 4: explain model
    shap_values = explain_model(optimized_model, X_train)

    return {

        "model_name": best_name,

        "accuracy": round(optimized_score, 4),

        "model": optimized_model,

        "shap_values": shap_values

    }
  
