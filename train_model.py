import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pipeline.build_dataset import build_ml_dataset
from pipeline.model_validation import evaluate_model_with_rolling_window

#Monta o dataset final
data = build_ml_dataset(
    "data/portfolio.json",
    "data/profile.json",
    "data/transcript.json"
)

#Variáveis explicativas
categ_vals = ['gender', 'offer_type']
numeric_vals = ['received_time', 'reward_y', 'difficulty', 'duration', 'pre_offer_avg_spend', 'pre_offer_num_tx', 'time_last_tx', 'membership_year', 'age'
                , 'income', 'spend_reward_ratio', 'spend_difficulty_ratio']
extra_vals = ['agi_missing', 'channel_mobile', 'channel_social', 'channel_web']

target = "offer_completed"

#Resultados para kNN
k_values = [5, 7, 10]
results_dict = {}

for k in k_values:
    print(f"\nAvaliando KNN com k = {k}")
    model = KNeighborsClassifier(n_neighbors=k, weights='distance')

    results = evaluate_model_with_rolling_window(
        df=data,
        categorical_features=categ_vals,
        numeric_features=numeric_vals,
        extra_features=extra_vals,
        target_col=target,
        base_model=model
    )

    results_dict[k] = results

summary = pd.DataFrame({
    "k": [5, 7, 10],
    "Acurácia Média": [np.mean(results_dict[k][0]) for k in [5,7,10]],
    "Precisão Média": [np.mean(results_dict[k][1]) for k in [5,7,10]],
    "Recall Médio": [np.mean(results_dict[k][2]) for k in [5,7,10]],
    "F1 Médio": [np.mean(results_dict[k][3]) for k in [5,7,10]],
    "F2 Médio": [np.mean(results_dict[k][4]) for k in [5,7,10]],
    "F3 Médio": [np.mean(results_dict[k][5]) for k in [5,7,10]],
    "AUC Média": [np.mean(results_dict[k][6]) for k in [5,7,10]],
})

print(summary)