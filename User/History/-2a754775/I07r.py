import polygon

forex_client = polygon.ForexClient('uy_C6zA5nsI2p9kX2I_bhsWGIAKEOyXL')  # for usual sync client
data = forex_client.get_full_range_aggregate_bars('EURUSD', '2010-06-28', '2022-06-28')
