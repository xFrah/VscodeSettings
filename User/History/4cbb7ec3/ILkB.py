import pandas as pd 

# read .pandas.dataframe
df = pd.read_pickle(r"C:\Users\fdimo\Downloads\China_A_shares.pandas.dataframe")
# get list of columns
print(df.columns)