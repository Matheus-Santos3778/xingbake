from pipeline.data_loading import load_datasets
from pipeline.data_preprocessing import expand_transcript, merge_with_portfolio, extract_transactions
from pipeline.feature_engineering import add_target_column, extract_membership_year, encode_channels, calculate_pre_offer_features, building_ratios
import pandas as pd
import numpy as np

def build_ml_dataset(portfolio_path, profile_path, transcript_path):
    
    #Carregamento de dados
    portfolio, profile, transcript = load_datasets(portfolio_path, profile_path, transcript_path)

    #Pré-processamento
    transcript_exp = expand_transcript(transcript)
    raw_data = merge_with_portfolio(transcript_exp, portfolio)

    #Base de ofertas recebidas
    offers = raw_data[raw_data['event'] == 'offer received'].copy()
    offers = offers.rename(columns={'time': 'received_time'})

    #Transações
    transactions = extract_transactions(transcript_exp)

    #Criação de features
    agg = calculate_pre_offer_features(offers, transactions)
    offers = offers.merge(agg, on=['person', 'offer_id', 'received_time'], how='left')

    #Caso não tenha transações anteriores, a média de gasto, número total de transações = 0, para time_last_tx usamos = -1 para identificar que não ocorreu
    offers['pre_offer_avg_spend'] = offers['pre_offer_avg_spend'].fillna(0)
    offers['pre_offer_num_tx'] = offers['pre_offer_num_tx'].fillna(0)
    offers['time_last_tx'] = offers['time_last_tx'].fillna(-1)

    #Incluindo variável target
    offers = add_target_column(offers, raw_data)

    #Unindo com a base profile
    data = pd.merge(offers, profile, how='left', left_on='person', right_on='id')

    #Extraindo a variável 'ano' de became_membership
    data = extract_membership_year(data)

    #Encoding de channels
    data = encode_channels(data)

    #Criando variáveis spend_reward_ratio e spend_difficulty_ratio
    data = building_ratios(data)

    #Criando variável indicadora de missing em Age, Gender e Income
    data['agi_missing'] = data['income'].isna().astype(int)

    #Limpeza de variáveis que repetidas/que não serão mais utilizadas
    drop_cols = ['event', 'offer_id', 'person', 'last_tx_time', 'amount', 'reward_x', 'id_x', 'id_y', 'offer id']
    data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

    return data