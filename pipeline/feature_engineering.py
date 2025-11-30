import pandas as pd
import numpy as np

#Criação das variáveis com informações do cliente até o tempo da oferta
def calculate_pre_offer_features(offers, transactions):

    transactions = transactions.rename(columns={'time': 'tx_time'})

    #Repete cada linha de oferta para todas as transações da mesma pessoa
    merged = offers[['person', 'offer_id', 'received_time']].merge(
        transactions,
        on='person',
        how='left'
    )

    #Filtra apenas transações que ocorreram antes da oferta
    merged = merged[merged['tx_time'] < merged['received_time']]

    #Agrupa e calcula as variáveis
    agg = merged.groupby(['person', 'offer_id', 'received_time']).agg(
        pre_offer_avg_spend=('amount', 'mean'),
        pre_offer_num_tx=('amount', 'count'),
        last_tx_time=('tx_time', 'max')
    ).reset_index()

    #Calcula tempo desde última transação
    agg['time_last_tx'] = agg['received_time'] - agg['last_tx_time']
    agg.drop(columns='last_tx_time', inplace=True)

    return agg

#Incluindo a variável resposta
def add_target_column(offers_df, raw_data):

    offers = offers_df.copy()
    completed = raw_data[raw_data['event'] == 'offer completed'].copy()
    completed = completed.rename(columns={'time': 'completed_time'})

    offers['valid_until'] = offers['received_time'] + offers['duration']

    #Unindo as bases de offer
    merged = offers.merge(
        completed[['person', 'offer_id', 'completed_time']],
        on=['person', 'offer_id'],
        how='left'
    )

    #Filtrando completed_times no intervalo
    valid_completions = merged[
        (merged['completed_time'] >= merged['received_time']) &
        (merged['completed_time'] <= merged['valid_until'])
    ]

    #Agrupa por pessoa + oferta recebida e pega o primeiro completed_time
    min_completed = valid_completions.groupby(
        ['person', 'offer_id', 'received_time']
    )['completed_time'].min().reset_index()

    #Junta novamente na base original
    offers = offers.merge(
        min_completed,
        on=['person', 'offer_id', 'received_time'],
        how='left'
    )

    offers['offer_completed'] = offers['completed_time'].notna().astype(int)
    offers.drop(['valid_until', 'completed_time'], axis=1, inplace=True)

    return offers

#Transformando a coluna channels em variáveis dummies
def encode_channels(df):

    exploded = df[['channels']].explode('channels')
    dummies = pd.get_dummies(exploded['channels'], prefix='channel')
    dummies_grouped = dummies.groupby(exploded.index).sum()

    df = pd.concat([df.drop(columns='channels'), dummies_grouped], axis=1)
    #Todas as ofertas usam o channel_email, então ela é uma preditora inútil
    df.drop('channel_email', axis=1, inplace=True) 
    
    return df

#Extraindo a variável ano
def extract_membership_year(df, date_col='became_member_on'):

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    df['membership_year'] = df[date_col].dt.year
    
    df.drop([date_col], axis=1, inplace=True)

    return df

def building_ratios(df):

    #Evita divisão por zero substituindo 0 por NaN
    safe_reward = df['reward_y'].replace(0, np.nan)
    safe_difficulty = df['difficulty'].replace(0, np.nan)

    df['spend_reward_ratio'] = np.log1p(df['pre_offer_avg_spend'] / safe_reward)
    df['spend_difficulty_ratio'] = np.log1p(df['pre_offer_avg_spend'] / safe_difficulty)

    #Substitui NaNs e infinitos restantes por 0
    df[['spend_reward_ratio', 'spend_difficulty_ratio']] = (
        df[['spend_reward_ratio', 'spend_difficulty_ratio']]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df