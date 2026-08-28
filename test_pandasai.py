import os
from dotenv import load_dotenv

import pandasai as pai
from pandasai_litellm.litellm import LiteLLM


load_dotenv()

llm = LiteLLM(
    model="groq/openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

pai.config.set({
    "llm": llm
})

df = pai.read_csv("data/receipts.csv")

answer = df.chat("What is the most expensive item I have purchased?")


print(answer)
