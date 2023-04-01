def fields2row(fields: dict):
    if fields == 0:
        return {}
    row = {f.get("name", None): f.get("value", None) for f in fields}
    return row
