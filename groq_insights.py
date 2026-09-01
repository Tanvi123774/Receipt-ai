import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

df = pd.read_csv("data/receipts.csv")

total_spending = df["Price"].sum()

print("Total spending:", round(total_spending, 2))
category_spending = df.groupby("Category")["Price"].sum()

print("\nSpending by category:")
print(category_spending)
prompt = f"""
You are a grocery spending assistant.

Here is the user's spending data:

Total spending: ₹{round(total_spending, 2)}

Spending by category:
{category_spending.to_string()}

Analyze this data.

Give:
1. A short spending summary
2. The highest spending category
3. Any noticeable spending pattern
4. Three simple everyday budgeting suggestions

Do not give investment advice.
Do not recalculate the numbers. Use the values provided.
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

insights = response.choices[0].message.content

print("\nGroq AI Insights:")
print(insights)