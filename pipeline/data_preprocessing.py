import pandas as pd

#Expande a coluna Value para as colunas offer_id e amount
def expand_transcript(transcript):

    df = transcript.copy()
    value_expanded = pd.json_normalize(df['value'])

    df = pd.concat([df.drop(columns='value'), value_expanded], axis=1)
    df['offer_id'] = df.get('offer id', pd.NA).fillna(df.get('offer_id'))

    return df

#Unindo a base transcript com a base portfolio
def merge_with_portfolio(transcript_exp, portfolio):

    df = pd.merge(transcript_exp, portfolio, left_on='offer_id', right_on='id', how='left')
    df['time'] = df['time'] / 24  #Converte para dias

    return df