import polygon

forex_client = polygon.ForexClient('uy_C6zA5nsI2p9kX2I_bhsWGIAKEOyXL')  # for usual sync client
data = forex_client.get_full_range_aggregate_bars('EURUSD', '2009-06-28', '2023-08-10')

print(data)