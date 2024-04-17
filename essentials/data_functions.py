import pandas as pd


def read_data(path: str, format : str = 'jsonl', **kwargs) -> pd.DataFrame:
    """Reads training data from github."""
    if format == 'jsonl':
        return pd.read_json(path, lines=True, **kwargs)
    elif format == 'csv':
        return pd.read_csv(path, **kwargs)
    else:
        raise TypeError