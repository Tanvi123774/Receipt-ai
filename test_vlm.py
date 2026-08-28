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
# LOAD API KEY
# =========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()


# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6e6e6;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION CACHE
# =========================================================

if "receipt_cache" not in st.session_state:
    st.session_state.receipt_cache = {}


# =========================================================
# GROQ LLM FOR PANDASAI
# =========================================================

groq_llm = GroqLLM(
    api_token=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================================================
# HEADER
# =========================================================

st.title("AI Grocery Receipt & Spending Analyst")

st.caption(
    "Upload multiple grocery receipts, extract the details automatically, "
    "analyze your monthly spending, and ask questions about your purchases."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.subheader("Upload receipts")

uploaded_files = st.file_uploader(
    "Upload one or more receipt images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# =========================================================
# NO FILES
# =========================================================

if not uploaded_files:
    st.info("Upload your monthly receipt images to begin.")
    st.stop()


# =========================================================
# PROCESS RECEIPTS
# =========================================================

all_items = []
processed_receipts = []

with st.spinner("Reading your receipts..."):

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        # =================================================
        # USE CACHE IF ALREADY PROCESSED
        # =================================================

        if file_hash in st.session_state.receipt_cache:

            receipt_data = st.session_state.receipt_cache[
                file_hash
            ]


        # =================================================
        # OTHERWISE CALL VLM
        # =================================================

        else:

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            temp_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension
                ) as temp_file:

                    temp_file.write(
                        file_bytes
                    )

                    temp_path = temp_file.name


                # =========================================
                # VLM CALL
                # =========================================

                receipt_data = extract_receipt_data(
                    temp_path
                )


                # =========================================
                # SAVE TO SESSION CACHE
                # =========================================

                st.session_state.receipt_cache[
                    file_hash
                ] = receipt_data


            except Exception as e:

                st.error(
                    f"Could not process {uploaded_file.name}"
                )

                st.code(
                    str(e)
                )

                continue


            finally:

                if temp_path and os.path.exists(temp_path):

                    try:
                        os.remove(temp_path)

                    except OSError:
                        pass


        # =================================================
        # STORE RECEIPT RESULT
        # =================================================

        processed_receipts.append(
            {
                "filename": uploaded_file.name,
                "data": receipt_data
            }
        )


        # =================================================
        # ADD ITEMS
        # =================================================

        items = receipt_data.get(
            "items",
            []
        )

        for item in items:

            if not isinstance(
                item,
                dict
            ):
                continue


            all_items.append(
                {
                    "Date": receipt_data.get("date"),
                    "Store": receipt_data.get("store"),
                    "Item": item.get("item"),
                    "Quantity": item.get("quantity"),
                    "Price": item.get("price")
                }
            )


# =========================================================
# CHECK DATA
# =========================================================

if not all_items:

    st.warning(
        "The uploaded receipts were processed, "
        "but no grocery items could be extracted."
    )

    st.stop()


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    all_items
)


# =========================================================
# CLEAN DATA
# =========================================================

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)


# =========================================================
# BASIC CALCULATIONS
# =========================================================

monthly_total = df["Price"].sum()

receipt_count = len(
    processed_receipts
)

purchase_count = len(
    df
)


if df["Price"].notna().any():

    expensive_row = df.loc[
        df["Price"].idxmax()
    ]

    expensive_item = expensive_row["Item"]

    expensive_price = expensive_row["Price"]

else:

    expensive_item = "Not available"

    expensive_price = 0


# =========================================================
# SUMMARY
# =========================================================

st.divider()

st.subheader("Monthly summary")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total spent",
        f"₹{monthly_total:,.2f}"
    )


with col2:

    st.metric(
        "Highest purchase",
        f"₹{expensive_price:,.2f}",
        help=str(expensive_item)
    )


with col3:

    st.metric(
        "Receipts uploaded",
        receipt_count,
        help=f"{purchase_count} items detected"
    )


# =========================================================
# ANALYSIS SECTION
# =========================================================

st.divider()

left, right = st.columns(
    2,
    gap="large"
)


# =========================================================
# SPENDING BY STORE
# =========================================================

with left:

    st.subheader(
        "Spending by store"
    )

    if df["Store"].notna().any():

        store_spending = (
            df
            .dropna(
                subset=["Store"]
            )
            .groupby(
                "Store"
            )["Price"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            store_spending
        )

    else:

        st.info(
            "Store names were not detected."
        )


# =========================================================
# TOP PURCHASES
# =========================================================

with right:

    st.subheader(
        "Highest purchases"
    )

    top_items = (
        df
        .dropna(
            subset=["Price"]
        )
        .sort_values(
            "Price",
            ascending=False
        )
        .head(5)
    )

    if not top_items.empty:

        st.dataframe(
            top_items[
                [
                    "Item",
                    "Store",
                    "Price"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ALL PURCHASES
# =========================================================

st.divider()

st.subheader(
    "All purchases"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# RAW RECEIPT DATA
# =========================================================

with st.expander(
    "View extracted receipt details"
):

    for receipt in processed_receipts:

        st.write(
            receipt["filename"]
        )

        st.json(
            receipt["data"]
        )


# =========================================================
# PANDASAI SMART DATAFRAME
# =========================================================

sdf = SmartDataframe(
    df,
    config={
        "llm": groq_llm,
        "verbose": False,
        "enable_cache": False
    }
)


# =========================================================
# ASK QUESTIONS
# =========================================================

st.divider()

st.subheader(
    "Ask about your spending"
)

st.caption(
    "Ask a question about all uploaded receipts."
)


question_col, button_col = st.columns(
    [5, 1]
)


with question_col:

    user_question = st.text_input(
        "Question",
        placeholder="Example: Which store did I spend the most at?",
        label_visibility="collapsed"
    )


with button_col:

    ask_button = st.button(
        "Ask",
        use_container_width=True
    )


# =========================================================
# ANSWER
# =========================================================

if ask_button:

    if not user_question:

        st.warning(
            "Enter a question first."
        )

    else:

        try:

            with st.spinner(
                "Analyzing your spending..."
            ):

                answer = sdf.chat(
                    user_question
                )


            st.success(
                str(answer)
            )


        except Exception as e:

            st.error(
                "Could not answer the question."
            )

            st.code(
                str(e)
            )


# =========================================================
# DOWNLOAD CSV
# =========================================================

st.divider()

csv_data = df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download extracted data",
    data=csv_data,
    file_name="monthly_receipt_data.csv",
    mime="text/csv"
)


# =========================================================
# CLEAR CACHE BUTTON
# =========================================================

st.divider()

if st.button(
    "Clear processed receipts"
):

    st.session_state.receipt_cache = {}

    st.rerun()