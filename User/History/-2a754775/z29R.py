import polygon

forex_client = polygon.ForexClient('KEY')  # for usual sync client
forex_client.get_full_range_aggregate_bars('AMD', '2005-06-28', '2022-06-28')