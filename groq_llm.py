import os
import re
import json
from datetime import datetime
from typing import Optional

from groq import Groq
from pandasai.llm.base import LLM


class GroqLLM(LLM):

    model: str = "openai/gpt-oss-20b"
    temperature: float = 0
    max_tokens: int = 1000


    def __init__(
        self,
        api_token: Optional[str] = None,
        **kwargs
    ):

        self.api_token = (
            api_token
            or os.getenv("GROQ_API_KEY")
        )

        if not self.api_token:
            raise ValueError("Groq API key is required")

        self.model = kwargs.get(
            "model",
            self.model
        )

        self.temperature = kwargs.get(
            "temperature",
            self.temperature
        )

        self.max_tokens = kwargs.get(
            "max_tokens",
            self.max_tokens
        )

        self.client = Groq(
            api_key=self.api_token
        )


    # ==========================================
    # SAVE QUERY TO JSON
    # ==========================================

    def save_query_to_json(
        self,
        user_query,
        sql_query
    ):

        file_name = "queries.json"

        records = []

        if os.path.exists(file_name):

            try:

                with open(
                    file_name,
                    "r",
                    encoding="utf-8"
                ) as file:

                    records = json.load(file)

                if not isinstance(records, list):
                    records = []

            except (
                json.JSONDecodeError,
                OSError
            ):
                records = []


        new_record = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "user_query": user_query,
            "sql_query": sql_query
        }

        records.append(new_record)


        with open(
            file_name,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                records,
                file,
                indent=4
            )


        print("\nQuery saved to queries.json\n")


    # ==========================================
    # MAIN PANDASAI → GROQ CALL
    # ==========================================

    def call(
        self,
        instruction,
        context=None
    ):

        prompt = instruction.to_string()


        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        )


        generated_code = (
            response
            .choices[0]
            .message
            .content
        )


        print(
            "\n========== GENERATED CODE =========="
        )

        print(generated_code)

        print(
            "====================================\n"
        )


        # ==========================================
        # EXTRACT USER QUERY
        # ==========================================

        query_match = re.search(
            r"### QUERY\s*(.*?)\s*At the end",
            prompt,
            re.DOTALL
        )


        if query_match:

            user_query = (
                query_match
                .group(1)
                .strip()
            )

        else:

            user_query = "Unknown"


        # ==========================================
        # TRY FORMAT 1:
        # sql_query = """ SELECT ... """
        # ==========================================

        sql_match = re.search(
            r'sql_query\s*=\s*"""(.*?)"""',
            generated_code,
            re.DOTALL
        )


        # ==========================================
        # TRY FORMAT 2:
        # execute_sql_query(""" SELECT ... """)
        # ==========================================

        if not sql_match:

            sql_match = re.search(
                r'execute_sql_query\s*\(\s*"""(.*?)"""',
                generated_code,
                re.DOTALL
            )


        # ==========================================
        # SQL FOUND
        # ==========================================

        if sql_match:

            generated_sql = (
                sql_match
                .group(1)
                .strip()
            )


            print(
                "\n========== SQL QUERY =========="
            )

            print(generated_sql)

            print(
                "===============================\n"
            )


            self.save_query_to_json(
                user_query,
                generated_sql
            )


        # ==========================================
        # SQL NOT FOUND
        # ==========================================

        else:

            print(
                "\nNo SQL query found.\n"
            )


        return generated_code


    @property
    def type(self) -> str:

        return "groq"