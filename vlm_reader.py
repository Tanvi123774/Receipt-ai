import os
import base64
import json
import mimetypes
import re

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    # Streamlit Cloud fallback
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:
        api_key = None


if not api_key:

    raise ValueError(
        "GROQ_API_KEY not found."
    )


client = Groq(
    api_key=api_key
)


# =========================================================
# PARSE JSON SAFELY
# =========================================================

def extract_json_from_text(text):

    if not text:

        raise ValueError(
            "The VLM returned an empty response."
        )


    text = text.strip()


    # -----------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TRY DIRECT JSON PARSING
    # -----------------------------------------------------

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        pass


    # -----------------------------------------------------
    # TRY TO FIND JSON OBJECT
    # -----------------------------------------------------

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )


    if match:

        json_text = (
            match.group()
        )

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError:

            pass


    raise ValueError(
        "Could not convert VLM output into JSON.\n\n"
        f"Raw VLM output:\n{text}"
    )


# =========================================================
# RECEIPT READER
# =========================================================

def extract_receipt_data(image_path):


    # =====================================================
    # READ IMAGE
    # =====================================================

    with open(
        image_path,
        "rb"
    ) as image_file:

        image_bytes = (
            image_file.read()
        )


    # =====================================================
    # BASE64
    # =====================================================

    base64_image = (
        base64.b64encode(
            image_bytes
        )
        .decode(
            "utf-8"
        )
    )


    # =====================================================
    # IMAGE TYPE
    # =====================================================

    mime_type, _ = (
        mimetypes.guess_type(
            image_path
        )
    )


    if mime_type is None:

        mime_type = (
            "image/jpeg"
        )


    # =====================================================
    # VLM PROMPT
    # =====================================================

    prompt = """
Read this grocery receipt image very carefully.

Your task is to extract every purchased grocery item
and return structured JSON.

Extract:

- store name
- receipt date
- currency
- every purchased item name
- quantity of each item
- price of each item
- receipt total


Return ONLY valid JSON.

Use EXACTLY this structure:

{
    "store": null,
    "date": null,
    "currency": null,
    "items": [
        {
            "name": null,
            "quantity": null,
            "price": null
        }
    ],
    "total": null
}


IMPORTANT ITEM NAME RULES:

1. The key for the product name MUST be "name".

2. Read the actual product description printed on the receipt.

3. Do not replace a readable product name with "Unknown".

4. Receipt product names may be abbreviated.
   Preserve the readable abbreviation if necessary.

5. For example:

   "ORG BANANAS" should be returned as:
   "name": "ORG BANANAS"

   "2% MILK" should be returned as:
   "name": "2% MILK"

6. Do not treat subtotal, tax, total, payment,
   change, card number, discounts, or balance
   as purchased items.

7. Every purchased product should appear as a
   separate object in the items array.

8. If a product name is partially readable,
   return the readable text instead of null.

9. Only use null for the item name if the text
   is genuinely impossible to read.


QUANTITY RULES:

1. Quantity must be numeric.

2. If an explicit quantity is printed, use it.

3. If an item appears once and no quantity is
   explicitly displayed, use 1.

4. Do not confuse item codes with quantities.


PRICE RULES:

1. Price must be numeric.

2. Do not include currency symbols inside
   the price value.

3. Use the final item price shown on the receipt.

4. Do not use subtotal or total as an item price.


CURRENCY RULES:

1. Detect the currency used on the receipt.

2. Return the currency symbol whenever possible.

3. Examples:

   "$" = US Dollar

   "₹" = Indian Rupee

   "€" = Euro

   "£" = British Pound

   "¥" = Japanese Yen

4. If the symbol is not visible, infer currency
   only if it is obvious from the receipt.

5. If currency cannot be determined, use null.


GENERAL RULES:

1. Do not return markdown.

2. Do not use ```json.

3. Do not include explanations.

4. Use valid JSON only.

5. Use double quotes around property names.

6. Do not use trailing commas.

7. Total must be numeric.

8. Never invent information that is not visible.

9. Start the response with {.

10. End the response with }.
"""


    # =====================================================
    # VLM CALL
    # =====================================================

    response = (
        client.chat.completions.create(

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
    )


    # =====================================================
    # GET RESPONSE
    # =====================================================

    result = (
        response
        .choices[0]
        .message
        .content
    )


    # =====================================================
    # PARSE JSON
    # =====================================================

    data = (
        extract_json_from_text(
            result
        )
    )


    # =====================================================
    # VALIDATE MAIN OBJECT
    # =====================================================

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "VLM response was not a dictionary."
        )


    data.setdefault(
        "store",
        None
    )

    data.setdefault(
        "date",
        None
    )

    data.setdefault(
        "currency",
        None
    )

    data.setdefault(
        "items",
        []
    )

    data.setdefault(
        "total",
        None
    )


    # =====================================================
    # VALIDATE ITEMS
    # =====================================================

    if not isinstance(
        data["items"],
        list
    ):

        data["items"] = []


    # =====================================================
    # NORMALIZE ITEM FORMAT
    # =====================================================
    #
    # This protects us even if the VLM returns:
    #
    # {"item": "Milk"}
    #
    # instead of:
    #
    # {"name": "Milk"}
    #
    # =====================================================

    cleaned_items = []


    for item in data["items"]:

        if not isinstance(
            item,
            dict
        ):

            continue


        # ---------------------------------------------
        # GET ITEM NAME
        # ---------------------------------------------

        item_name = (
            item.get("name")
            or item.get("item")
            or item.get("product")
            or item.get("product_name")
        )


        # ---------------------------------------------
        # GET QUANTITY
        # ---------------------------------------------

        quantity = (
            item.get(
                "quantity",
                1
            )
        )


        if quantity is None:

            quantity = 1


        # ---------------------------------------------
        # GET PRICE
        # ---------------------------------------------

        price = (
            item.get(
                "price",
                None
            )
        )


        # ---------------------------------------------
        # CLEAN ITEM
        # ---------------------------------------------

        cleaned_item = {

            "name": item_name,

            "quantity": quantity,

            "price": price
        }


        cleaned_items.append(
            cleaned_item
        )


    data["items"] = (
        cleaned_items
    )


    return data