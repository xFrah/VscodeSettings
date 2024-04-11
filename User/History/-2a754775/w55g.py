from datetime import datetime, timedelta
import time
from turtle import pd
from polygon import ForexClient
import pytz
import requests
import pickle

API_KEY = "uy_C6zA5nsI2p9kX2I_bhsWGIAKEOyXL"

timespan = "minute"
start_date = datetime(2021, 5, 12)
end_date = datetime(2023, 7, 12)

current_date = start_date
all_data = []

ticker = "C:EURUSD"

# # open pickle file
# with open("data2122.pickle", "rb") as f:
#     all_data = pickle.load(f)
#     print(len(all_data))
#     last_date = all_data[-1]["t"]
#     first_date = all_data[0]["t"]
#     first = datetime.fromtimestamp(first_date / 1000)
#     current_date = datetime.fromtimestamp(last_date / 1000) + timedelta(days=1)
#     print(current_date)

# while True:
#     pass

dataframes = []

while current_date < end_date:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date.date()}/{end_date.date()}?adjusted=true&sort=asc&limit=50000000&apiKey={API_KEY}"

    response = requests.get(url)

    data = response.json()

    results = data["results"]

    # get last date
    last_date = datetime.fromtimestamp(results[-1]["t"] / 1000)

    datetime.fromtimestamp(last_date / 1000)


df = pd.DataFrame(results)

df["t"] = pd.to_datetime(df["t"], unit="ms")

dataframes.append(df)

# serialize all_data
with open("data.pickle", "wb") as f:
    pickle.dump(all_data, f)
