import pandas as pd
import numpy as np
import os
from pipeline.build_dataset import build_ml_dataset
from pipeline.model_validation import evaluate_model_with_rolling_window
from itertools import product

import seaborn as sns                       
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

#Monta o dataset final
data = build_ml_dataset(
    "data/portfolio.json",
    "data/profile.json",
    "data/transcript.json"
)

#Parâmetros para fazer um 'grid search'
param_grids = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [5, 7, 10, 15],
            "weights": ["uniform", "distance"]
        }
    },

    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [None, 5, 10]
        }
    },

    "LogReg": {
        "model": LogisticRegression(max_iter=2000),
        "params": {
            "C": [0.1, 1, 10],
            "class_weight": [None, "balanced"]
        }
    },

    "GaussianNB": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },

    "LDA": {
        "model": LinearDiscriminantAnalysis(),
        "params": {
            "solver": ["svd", "lsqr"],
            "shrinkage": [None, 0.1, 0.3, 0.5]
        }
    },

    "QDA": {
        "model": QuadraticDiscriminantAnalysis(),
        "params": {
            "reg_param": [0.0, 0.01, 0.1, 0.3]
        }
    }
}

#Variáveis explicativas
categ_vals = ['gender', 'offer_type']
numeric_vals = ['received_time', 'reward_y', 'difficulty', 'duration', 'pre_offer_avg_spend', 'pre_offer_num_tx', 'time_last_tx', 'membership_year', 'age'
                , 'income', 'spend_reward_ratio', 'spend_difficulty_ratio']
extra_vals = ['agi_missing', 'channel_mobile', 'channel_social', 'channel_web']

target = "offer_completed"

#Função que treina todos modelos do grid search
def rolling_window_grid_search(df, categorical_features, numeric_features, extra_features, target_col, model, param_grid):  

    # gera todas as combinações possíveis
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = list(product(*values))

    best_score = -np.inf
    best_params = None
    best_results = None

    print(f"\n🔧 Testando {len(param_combinations)} combinações de parâmetros...")

    for combo in param_combinations:
        params = dict(zip(keys, combo))
        print(f"\n➡️ Testando params: {params}")

        model.set_params(**params)

        results = evaluate_model_with_rolling_window(
            df=df,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            extra_features=extra_features,
            target_col=target_col,
            base_model=model
        )

        #Métrica que estamos maximizando
        f2_mean = np.mean(results[4])
        print(f"   F2 Médio = {f2_mean:.4f}")

        if f2_mean > best_score:
            best_score = f2_mean
            best_params = params
            best_results = results

    return best_params, best_results, best_score

results_overall = []

for model_name, config in param_grids.items():
    print(f"\n============================")
    print(f"🔍 Iniciando Grid Search para: {model_name}")
    print(f"============================")

    best_params, best_results, best_score = rolling_window_grid_search(
        df=data,
        categorical_features=categ_vals,
        numeric_features=numeric_vals,
        extra_features=extra_vals,
        target_col=target,
        model=config["model"],
        param_grid=config["params"]
    )

    results_overall.append({
        "Modelo": model_name,
        "Melhores Parâmetros": best_params,
        "F2 Médio": best_score
    })

    #Visualização melhor modelo
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    aucs = best_results[6]
    conf_matrix = best_results[7]
    mean_tpr = best_results[8]
    std_tpr = best_results[9]
    mean_fpr = best_results[10]

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
    filepath = os.path.join("results", 'auc_cv_' + model_name)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

summary = pd.DataFrame(results_overall)
print("\n\n📊 Ranking final dos modelos:")
print(summary.sort_values("F2 Médio", ascending=False))