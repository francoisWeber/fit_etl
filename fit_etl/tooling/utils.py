import pandas as pd


def fields2row(fields: dict):
    if fields == 0:
        return {}
    row = {f.get("name", None): f.get("value", None) for f in fields}
    return row


def reset_time(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        Warning("Datafram has no date-based index !")
    return df.assign(elapsed_time=(df.index - df.index[0]).seconds).set_index(
        "elapsed_time"
    )
