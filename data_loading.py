import pandas as pd

def load_datasets(portfolio_path, profile_path, transcript_path):

    portfolio = pd.read_json(portfolio_path, orient='records', lines=True)
    profile = pd.read_json(profile_path, orient='records', lines=True)
    transcript = pd.read_json(transcript_path, orient='records', lines=True)
    
    return portfolio, profile, transcript