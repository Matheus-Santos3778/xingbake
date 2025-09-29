import pandas as pd

def extract_transactions(transcript_exp):

    df = transcript_exp[transcript_exp['event'] == 'transaction'].copy()
    df = df[['person', 'time', 'amount']]
    df['time'] = df['time'] / 24

    return df

#Criação das variáveis com informações do cliente até o tempo da oferta
def calculate_pre_offer_features(offers, transactions):

    offers_tx = offers[['person', 'offer_id', 'received_time']].copy()
    offers_tx = offers_tx.sort_values(['person', 'received_time']).copy()
    offers_tx['previous_received_time'] = offers_tx.groupby('person')['received_time'].shift(1)
    offers_tx['previous_received_time'] = offers_tx['previous_received_time'].fillna(-float('inf'))

    merged = transactions.merge(offers_tx, on='person', how='inner')
    mask = (merged['time'] >= merged['previous_received_time']) & (merged['time'] < merged['received_time'])
    merged = merged[mask]

    agg = merged.groupby(['person', 'offer_id', 'received_time']).agg(
        pre_offer_avg_spend=('amount', 'mean'),
        pre_offer_num_tx=('amount', 'count'),
        last_tx_time=('time', 'max')
    ).reset_index()

    agg['pre_offer_time_since_last_tx'] = agg['received_time'] - agg['last_tx_time']

    return agg