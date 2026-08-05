import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional


# ----------------------------
# Pydantic Models
# ----------------------------

class ProductPrice(BaseModel):
    store: str
    variant: Optional[str] = None
    price: str
    availability: Optional[str] = None
    url: Optional[str] = None


class Summary(BaseModel):
    lowest_price: str
    highest_price: str
    average_price: Optional[str] = None
    best_store: str


class ProductComparison(BaseModel):
    product_name: str
    price_comparison: List[ProductPrice]
    summary: Summary
    recommendation: str


# ----------------------------
# Model
# ----------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

structured_model = model.with_structured_output(ProductComparison)

# ----------------------------
# Search Tool
# ----------------------------

search_tool = TavilySearchResults(
    max_results=5
)

# ----------------------------
# Prompt
# ----------------------------

prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI Product Price Finder.

Analyze the following web search results.

Instructions:
- Identify the product name.
- Extract prices from different stores.
- Extract product variants if available.
- Extract availability.
- Extract product URLs.
- Ignore duplicate or unrelated results.
- If information is missing, leave the field empty.

Search Results:
{search_results}
""")

chain = prompt | structured_model

# ----------------------------
# UI
# ----------------------------

st.set_page_config(
    page_title="AI Product Price Finder",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 AI Product Price Finder")
st.caption("Compare prices from multiple stores using AI + Web Search")

product = st.text_input(
    "Search Product",
    placeholder="e.g. Samsung Galaxy S25 Ultra"
)

search = st.button("🔍 Search", use_container_width=True)

if search:

    if product.strip() == "":
        st.warning("Enter a product name.")
        st.stop()

    with st.spinner("Searching the web..."):

        search_results = search_tool.run(
            product + " price in Pakistan buy online"
        )

        response = chain.invoke(
            {
                "search_results": search_results
            }
        )

    st.success("Search Completed")

    st.header(response.product_name)

    # ----------------------------
    # Metrics
    # ----------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Lowest Price",
        response.summary.lowest_price
    )

    c2.metric(
        "Highest Price",
        response.summary.highest_price
    )

    c3.metric(
        "Average Price",
        response.summary.average_price or "N/A"
    )

    c4.metric(
        "Best Store",
        response.summary.best_store
    )

    st.divider()

    # ----------------------------
    # Price Table
    # ----------------------------

    st.subheader("Price Comparison")

    df = pd.DataFrame(
        [item.model_dump() for item in response.price_comparison]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------
    # Recommendation
    # ----------------------------

    st.subheader("Recommendation")

    st.info(response.recommendation)

