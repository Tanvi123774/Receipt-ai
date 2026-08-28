import os
import base64
import json
import mimetypes
import re

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


client = Groq(api_key=api_key)


# =========================================================
# CLEAN MODEL OUTPUT
# =========================================================

def extract_json_from_text(text):
    """
    Safely extracts JSON even if the model adds markdown
    such as ```json ... ```
    """

    text = text.strip()

    # Remove ```json and ```
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # First try normal JSON
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass


    # Try finding first JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        json_text = match.group()

        try:
            return json.loads(json_text)

        except json.JSONDecodeError:
            pass


    raise ValueError(
        f"Could not parse VLM response as JSON.\n\nRaw response:\n{text}"
    )


# =========================================================
# RECEIPT EXTRACTION
# =========================================================

def extract_receipt_data(image_path):

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    with open(image_path, "rb") as image_file:

        image_bytes = image_file.read()


    # -----------------------------------------------------
    # BASE64 ENCODE
    # -----------------------------------------------------

    base64_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")


    # -----------------------------------------------------
    # DETECT IMAGE TYPE
    # -----------------------------------------------------

    mime_type, _ = mimetypes.guess_type(
        image_path
    )

    if mime_type is None:
        mime_type = "image/jpeg"


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = """
You are reading a grocery shopping receipt.

Carefully inspect the receipt image and extract the information.

Return ONLY one valid JSON object.

Do not use markdown.
Do not use ```json.
Do not write explanations before or after the JSON.

Use this exact structure:

{
  "store": null,
  "date": null,
  "items": [
    {
      "item": null,
      "quantity": null,
      "price": null
    }
  ],
  "total": null
}

Rules:

1. "store" should contain the store name.
2. "date" should contain the receipt date if visible.
3. Every purchased product must be a separate object inside "items".
4. "item" should contain only the product name.
5. "quantity" should be a number if clearly visible.
6. "price" should be the final amount for that line item.
7. Prices must be JSON numbers, not strings.
8. Do not include currency symbols inside price values.
9. "total" must be the final receipt total as a number.
10. If a value cannot be clearly read, use null.
11. Never invent missing products or values.
12. Make sure all property names use double quotes.
13. Do not add trailing commas.
14. Your entire response must begin with { and end with }.
"""


    # -----------------------------------------------------
    # SEND TO VLM
    # -----------------------------------------------------

    response = client.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",
                        "text": prompt
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url":
                                f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],

        temperature=0,

        max_tokens=2000
    )


    # -----------------------------------------------------
    # GET MODEL RESPONSE
    # -----------------------------------------------------

    result = response.choices[0].message.content


    # -----------------------------------------------------
    # CONVERT TO PYTHON DICTIONARY
    # -----------------------------------------------------

    data = extract_json_from_text(
        result
    )


    # -----------------------------------------------------
    # SAFETY CHECKS
    # -----------------------------------------------------

    if not isinstance(data, dict):

        raise ValueError(
            "VLM returned data that was not a JSON object."
        )


    if "items" not in data:

        data["items"] = []


    if not isinstance(
        data["items"],
        list
    ):

        data["items"] = []


    # -----------------------------------------------------
    # ENSURE EXPECTED FIELDS
    # -----------------------------------------------------

    data.setdefault(
        "store",
        None
    )

    data.setdefault(
        "date",
        None
    )

    data.setdefault(
        "total",
        None
    )


    return data