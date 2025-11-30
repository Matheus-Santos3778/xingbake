import seaborn as sns                       
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, fbeta_score)
from sklearn.base import clone
import numpy as np  

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#Função de avaliação dos modelos e validação rolling window
def evaluate_model_with_rolling_window(
    df,
    categorical_features,
    numeric_features,
    extra_features,
    target_col,
    base_model
):

    df = df.copy()
    df = df.sort_values("received_time")

    #Splits por tempos de 'received_time'
    batch_times = sorted(df["received_time"].unique())
    splits = []
    for i in range(1, len(batch_times)):
        train_times = batch_times[:i]
        test_time = batch_times[i]

        train_idx = df["received_time"].isin(train_times)
        test_idx = df["received_time"] == test_time

        splits.append((train_idx, test_idx))

    #Pipeline
    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("income_imp", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), #Imputamos o income também com a mediana
            ("scaler", StandardScaler())
        ]), ["income"]),
        ("age_imp", Pipeline([
            ("replace_118", FunctionTransformer(
                lambda x: np.where(x == 118, np.nan, x), validate=False #Idade imputamos as que são 118 com a mediana
            )),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), ["age"]),
        ("num_no_imp", Pipeline([
            ("scaler", StandardScaler())
        ]), [
            col for col in numeric_features 
            if col not in ["age", "income"]
        ]),
        ("extra", "passthrough", extra_features)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", base_model)
    ])

    #Métricas agregadas
    accs, precisions, recalls, f1s = [], [], [], []
    f2s, f3s, aucs = [], [], []

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    #Loop das rolling windows
    for train_idx, test_idx in splits:
        X_train = df.loc[train_idx, categorical_features + numeric_features + extra_features]
        y_train = df.loc[train_idx, target_col]

        X_test = df.loc[test_idx, categorical_features + numeric_features + extra_features]
        y_test = df.loc[test_idx, target_col]

        model_copy = clone(pipeline)
        model_copy.fit(X_train, y_train)

        #Predições
        y_pred = model_copy.predict(X_test)

        if hasattr(model_copy["model"], "predict_proba"):
            y_prob = model_copy.predict_proba(X_test)[:, 1]
        else:
            s = model_copy.decision_function(X_test)
            y_prob = (s - s.min()) / (s.max() - s.min())

        #Métricas
        accs.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        f2s.append(fbeta_score(y_test, y_pred, beta=2))
        f3s.append(fbeta_score(y_test, y_pred, beta=3))
        aucs.append(roc_auc_score(y_test, y_prob))

        #Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    #Visualização

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    conf_matrix = confusion_matrix(y_true_all, y_pred_all)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(mean_fpr, mean_tpr, label=f"ROC Média (AUC = {np.mean(aucs):.3f})")
    axes[0].fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                         np.minimum(mean_tpr + std_tpr, 1), alpha=0.2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].legend()
    axes[0].set_title("Curva ROC - Rolling Window")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Não", "Sim"], yticklabels=["Não", "Sim"],
                ax=axes[1])
    axes[1].set_title("Matriz de Confusão Agregada")

    plt.tight_layout()
    plt.show()

    return accs, precisions, recalls, f1s, f2s, f3s, aucs