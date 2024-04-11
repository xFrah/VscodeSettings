import polygon
import pandas as pd


forex_client = polygon.ForexClient("uy_C6zA5nsI2p9kX2I_bhsWGIAKEOyXL")  # for usual sync client
data = forex_client.get_full_range_aggregate_bars("EURUSD", "2009-08-28", "2023-08-10")

df = pd.DataFrame(data)

# normalize date
df["t"] = pd.to_datetime(df["t"], unit="ms")

print(df)
