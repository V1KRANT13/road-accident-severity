import streamlit as st
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 42

DROP_COLUMNS = [
    "Casualty_severity",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Number_of_casualties",
]

st.set_page_config(page_title="Road Accident Severity Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Road Accident Severity Prediction")
st.write("Train a model on the road accident dataset and predict the likely accident severity.")

uploaded_file = st.file_uploader("Upload RTA Dataset.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload your `RTA Dataset.csv` file to train the model and use the prediction form.")
    st.stop()

@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(file_bytes):
    from io import BytesIO

    data = pd.read_csv(BytesIO(file_bytes))
    data.columns = data.columns.str.strip()
    data = data.replace(r"^\s*\?\s*$", np.nan, regex=True)
    data = data.drop_duplicates()

    if "Accident_severity" not in data.columns:
        raise ValueError("The CSV must contain an 'Accident_severity' target column.")

    data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    if "Time" in data.columns:
        data["Hour"] = pd.to_datetime(
            data["Time"], format="%H:%M:%S", errors="coerce"
        ).dt.hour
        data = data.drop(columns=["Time"])

    data = data.dropna(subset=["Accident_severity"])
    data["Accident_severity"] = data["Accident_severity"].astype(str).str.strip()

    X = data.drop(columns=["Accident_severity"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["Accident_severity"])

    if len(encoder.classes_) < 2:
        raise ValueError("The target must contain at least two severity classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    scores = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        fitted[name] = pipeline
        scores.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Macro F1": f1_score(y_test, pred, average="macro")
        })

    score_df = pd.DataFrame(scores).sort_values("Macro F1", ascending=False)
    best_name = score_df.iloc[0]["Model"]

    # Refit the selected model on all available labelled data for deployment.
    best_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", candidates[best_name])
    ])
    best_model.fit(X, y)

    # Keep holdout evaluation for honest reporting.
    best_test_model = fitted[best_name]
    best_pred = best_test_model.predict(X_test)
    report = classification_report(
        y_test, best_pred, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, output_dict=True, zero_division=0
    )

    return {
        "model": best_model,
        "scores": score_df,
        "best_name": best_name,
        "classes": encoder.classes_,
        "report": report,
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": best_pred,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

try:
    bundle = train_models(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

st.success("Model training completed.")

m1, m2 = st.columns(2)
m1.metric("Selected model", bundle["best_name"])
m2.metric("Test Macro F1", f'{bundle["scores"].iloc[0]["Macro F1"]:.3f}')

with st.expander("Model comparison", expanded=True):
    st.dataframe(bundle["scores"].round(4), use_container_width=True)

with st.expander("Test-set classification report"):
    st.dataframe(pd.DataFrame(bundle["report"]).T.round(3), use_container_width=True)

st.subheader("Predict accident severity")
st.caption("Enter the accident conditions below. The model uses the same features and preprocessing as training.")

X = bundle["X"]
values = {}

with st.form("prediction_form"):
    columns = st.columns(2)
    for idx, feature in enumerate(X.columns):
        col = columns[idx % 2]
        series = X[feature]
        if feature in bundle["numeric_cols"]:
            numeric = pd.to_numeric(series, errors="coerce")
            default = float(numeric.median()) if numeric.notna().any() else 0.0
            minimum = float(numeric.min()) if numeric.notna().any() else default - 1
            maximum = float(numeric.max()) if numeric.notna().any() else default + 1
            if minimum == maximum:
                minimum, maximum = default - 1, default + 1
            values[feature] = col.number_input(
                str(feature), min_value=minimum, max_value=maximum,
                value=min(max(default, minimum), maximum)
            )
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            options = options or ["Unknown"]
            values[feature] = col.selectbox(str(feature), options)

    submitted = st.form_submit_button("Predict severity", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=X.columns)
    prediction = bundle["model"].predict(input_df)[0]
    predicted_class = bundle["classes"][prediction]

    st.subheader("Prediction")
    st.success(f"Predicted accident severity: **{predicted_class}**")

    if hasattr(bundle["model"], "predict_proba"):
        probabilities = bundle["model"].predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Severity": bundle["classes"],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.write("Class probabilities")
        st.bar_chart(probability_df.set_index("Severity"))
