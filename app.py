import streamlit as st
import pandas as pd
import tempfile
import os
import hashlib

from dotenv import load_dotenv
from pandasai import SmartDataframe

from vlm_reader import extract_receipt_data
from groq_llm import GroqLLM


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Grocery Receipt & Spending Analyst",
    page_icon="🧾",
    layout="wide"
)


# =========================================================
# LOAD GROQ API KEY
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        padding: 22px !important;
        border-radius: 14px !important;
        min-height: 130px;
    }

    [data-testid="stMetricLabel"] p {
        color: #444444 !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }

    [data-testid="stMetricValue"] * {
        color: #111111 !important;
    }

    [data-testid="stMetric"] svg {
        color: #555555 !important;
        fill: #555555 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.title("🧾 AI Grocery Receipt & Spending Analyst")

st.write(
    "Upload your grocery receipts and analyze your spending using AI."
)


# =========================================================
# SESSION CACHE
# =========================================================

if "receipt_cache" not in st.session_state:
    st.session_state.receipt_cache = {}


# =========================================================
# CREATE GROQ LLM FOR PANDASAI
# =========================================================

groq_llm = GroqLLM(
    api_token=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================================================
# AUTOMATIC ITEM CATEGORIZATION
# =========================================================

def categorize_item(item_name):

    item = str(item_name).lower()

    categories = {

        "Dairy": [
            "milk",
            "cheese",
            "butter",
            "yogurt",
            "curd",
            "cream",
            "paneer"
        ],

        "Fruits": [
            "apple",
            "banana",
            "orange",
            "mango",
            "grape",
            "watermelon",
            "papaya",
            "pineapple",
            "strawberry"
        ],

        "Vegetables": [
            "tomato",
            "potato",
            "onion",
            "spinach",
            "cucumber",
            "cabbage",
            "carrot",
            "broccoli",
            "pepper"
        ],

        "Snacks": [
            "chips",
            "chocolate",
            "biscuit",
            "cookie",
            "cereal",
            "cracker",
            "popcorn",
            "snack"
        ],

        "Beverages": [
            "juice",
            "coffee",
            "tea",
            "soda",
            "cola",
            "drink",
            "water"
        ],

        "Protein": [
            "egg",
            "chicken",
            "fish",
            "meat",
            "beef",
            "mutton"
        ],

        "Grains": [
            "rice",
            "wheat",
            "bread",
            "flour",
            "oats"
        ],

        "Pulses": [
            "dal",
            "lentil",
            "beans",
            "chickpea"
        ],

        "Nuts": [
            "almond",
            "cashew",
            "peanut",
            "walnut",
            "pistachio"
        ],

        "Household": [
            "soap",
            "detergent",
            "cleaner",
            "tissue",
            "toilet",
            "shampoo"
        ]
    }

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in item:
                return category

    return "Other"


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "Upload grocery receipts",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# =========================================================
# PROCESS RECEIPTS
# =========================================================

if uploaded_files:

    receipts = []

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        # -------------------------------------------------
        # USE CACHE IF RECEIPT WAS ALREADY PROCESSED
        # -------------------------------------------------

        if file_hash in st.session_state.receipt_cache:

            receipt_data = (
                st.session_state
                .receipt_cache[file_hash]
            )

        else:

            # ---------------------------------------------
            # SAVE TEMPORARY IMAGE
            # ---------------------------------------------

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    file_bytes
                )

                temp_path = (
                    temp_file.name
                )


            # ---------------------------------------------
            # SEND RECEIPT TO VLM
            # ---------------------------------------------

            try:

                with st.spinner(
                    f"Reading {uploaded_file.name}..."
                ):

                    receipt_data = (
                        extract_receipt_data(
                            temp_path
                        )
                    )

                st.session_state.receipt_cache[
                    file_hash
                ] = receipt_data

            except Exception as e:

                st.error(
                    f"Could not process "
                    f"{uploaded_file.name}: {e}"
                )

                receipt_data = None

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)


        if receipt_data:
            receipts.append(
                receipt_data
            )


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    rows = []

    for receipt in receipts:

        store = receipt.get(
            "store",
            "Unknown"
        )

        date = receipt.get(
            "date",
            "Unknown"
        )

        currency = receipt.get(
            "currency",
            ""
        )


        for item in receipt.get(
            "items",
            []
        ):

            item_name = item.get(
                "name",
                "Unknown"
            )

            rows.append(
                {
                    "Date": date,

                    "Store": store,

                    "Currency": currency,

                    "Item": item_name,

                    "Category": categorize_item(
                        item_name
                    ),

                    "Quantity": item.get(
                        "quantity",
                        1
                    ),

                    "Price": item.get(
                        "price",
                        0
                    )
                }
            )


    df = pd.DataFrame(rows)


    # =====================================================
    # CHECK DATA
    # =====================================================

    if df.empty:

        st.warning(
            "No grocery items were detected."
        )

        st.stop()


    # =====================================================
    # CLEAN NUMERIC COLUMNS
    # =====================================================

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    ).fillna(1)


    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    ).fillna(0)


    # =====================================================
    # NORMALIZED ITEM
    # =====================================================

    df["Normalized_Item"] = (
        df["Item"]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    # =====================================================
    # DETECT CURRENCY
    # =====================================================

    currencies = (
        df["Currency"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    currencies = [
        currency
        for currency in currencies.unique()
        if currency
    ]


    if len(currencies) == 1:

        currency_symbol = currencies[0]

    elif len(currencies) == 0:

        currency_symbol = ""

    else:

        currency_symbol = None


    if currency_symbol is None:

        st.warning(
            "Multiple currencies were detected. "
            "Combined spending totals are hidden because "
            "different currencies should not be added together."
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader("Spending Summary")


    metric1, metric2, metric3 = st.columns(3)


    # -----------------------------------------------------
    # TOTAL SPENDING
    # -----------------------------------------------------

    if currency_symbol is not None:

        total_spending = df["Price"].sum()

        metric1.metric(
            "Total spent",
            f"{currency_symbol}{total_spending:.2f}"
        )

    else:

        metric1.metric(
            "Total spent",
            "Multiple currencies"
        )


    # -----------------------------------------------------
    # HIGHEST PURCHASE
    # -----------------------------------------------------

    highest_index = (
        df["Price"]
        .idxmax()
    )

    highest_row = (
        df.loc[highest_index]
    )

    highest_currency = str(
        highest_row.get(
            "Currency",
            ""
        )
    )

    metric2.metric(
        "Highest purchase",
        (
            f"{highest_row['Item']} "
            f"({highest_currency}"
            f"{highest_row['Price']:.2f})"
        )
    )


    # -----------------------------------------------------
    # NUMBER OF RECEIPTS
    # -----------------------------------------------------

    metric3.metric(
        "Receipts uploaded",
        len(receipts)
    )


    st.divider()


    # =====================================================
    # CHARTS
    # =====================================================

    if currency_symbol is not None:

        left, right = st.columns(2)


        # -------------------------------------------------
        # SPENDING BY CATEGORY
        # -------------------------------------------------

        with left:

            st.subheader(
                "Spending by category"
            )

            category_spending = (
                df.groupby("Category")["Price"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                category_spending
            )


        # -------------------------------------------------
        # SPENDING BY STORE
        # -------------------------------------------------

        with right:

            st.subheader(
                "Spending by store"
            )

            store_spending = (
                df.groupby("Store")["Price"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                store_spending
            )


    # =====================================================
    # ALL PURCHASES
    # =====================================================

    st.subheader("All purchases")


    display_df = df.drop(
        columns=["Normalized_Item"]
    )


    st.dataframe(
        display_df,
        width="stretch"
    )


    # =====================================================
    # RAW RECEIPT DETAILS
    # =====================================================

    with st.expander(
        "View extracted receipt details"
    ):

        for index, receipt in enumerate(
            receipts,
            start=1
        ):

            st.write(
                f"Receipt {index}"
            )

            st.json(
                receipt
            )


    st.divider()


    # =====================================================
    # AI CHATBOT
    # =====================================================

    st.subheader(
        "Ask questions about your spending"
    )


    sdf = SmartDataframe(
        df,
        config={
            "llm": groq_llm,
            "verbose": True,
            "enable_cache": False
        }
    )


    user_question = st.text_input(
        "Ask a question about your receipts"
    )


    if st.button(
        "Ask",
        width="stretch"
    ):

        if user_question.strip():

            try:

                # =========================================
                # ADD STORE + DATE TO GENERATED SQL
                # =========================================

                internal_question = f"""
{user_question}

IMPORTANT:
When generating the SQL query, if the answer comes from
specific receipt rows, include Store and Date in the
SELECT statement so that the source receipt can be
identified.

Do not remove Store or Date when selecting a specific
purchase or item.
"""


                with st.spinner(
                    "Analyzing..."
                ):

                    answer = sdf.chat(
                        internal_question
                    )


                st.success(
                    "Answer"
                )

                st.write(
                    answer
                )


            except Exception as e:

                st.error(
                    f"Could not answer question: {e}"
                )

        else:

            st.warning(
                "Enter a question first."
            )


    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    st.divider()

    st.subheader(
        "Export data"
    )


    csv_data = (
        display_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="grocery_spending.csv",
        mime="text/csv"
    )


    # =====================================================
    # CLEAR CACHE
    # =====================================================

    if st.button(
        "Clear processed receipts"
    ):

        st.session_state.receipt_cache = {}

        st.success(
            "Processed receipt cache cleared."
        )

        st.rerun()