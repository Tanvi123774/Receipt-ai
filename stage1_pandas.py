import pandas as pd

df = pd.read_csv("data/receipts.csv")

print(df)
print(df["Item"])
snacks = df[df["Category"] == "Dairy"] 

print(snacks)
total = df["Price"].sum()

print("Total spending:", total)
average = df["Price"].mean()

print("Average price:", average)
category_spending = df.groupby("Category")["Price"].sum()

print(category_spending)