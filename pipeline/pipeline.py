from data_loading import load_datasets
from data_preprocessing import expand_transcript, merge_with_portfolio
from feature_engineering import extract_transactions, calculate_pre_offer_features
import pandas as pd

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

    #Unindo com a base profile
    data = pd.merge(offers, profile, how='left', left_on='person', right_on='id')

    #Limpeza de variáveis que repetidas/que não serão mais utilizadas
    drop_cols = ['event', 'offer_id', 'person', 'last_tx_time', 'amount', 'reward_x', 'offer id', 'id_x', 'id_y']
    data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

    return data