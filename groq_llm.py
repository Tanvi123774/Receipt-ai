import os
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
        **kwargs,
    ):
        self.api_token = api_token or os.getenv("GROQ_API_KEY")

        if not self.api_token:
            raise ValueError("Groq API key is required")

        self.model = kwargs.get("model", self.model)
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

        return (
            response
            .choices[0]
            .message
            .content
        )

    @property
    def type(self) -> str:
        return "groq"