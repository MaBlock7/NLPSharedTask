import pandas as pd


def read_data(path: str, format : str = 'jsonl') -> pd.DataFrame:
    """Reads training data from github."""
    if format == 'jsonl':
        return pd.read_json(path, lines=True)
    elif format == 'csv':
        return pd.read_csv(path)
    else:
        raise TypeError