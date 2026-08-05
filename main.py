#this is newssumarizer tool
from dotenv import load_dotenv
load_dotenv( )
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional


class ProductPrice(BaseModel):
    store: str = Field(description="Store name")
    variant: Optional[str] = Field(default=None, description="Product variant")
    price: str = Field(description="Product price")
    availability: Optional[str] = Field(default=None, description="Availability")
    url: Optional[str] = Field(default=None, description="Product URL")


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



#model
model=ChatGoogleGenerativeAI(
model="gemini-2.5-flash"
)

#tool builtin
search_tool=TavilySearchResults(
    max_results=5
)

propmt_tempalte=ChatPromptTemplate.from_template(
"""
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
"""
)

parser=model.with_structured_output(ProductComparison)
chain=propmt_tempalte | parser

query=input("Which product Your are seraching :")
reuslt=search_tool.run(query+"price in Pakistan buy online")

response=chain.invoke({"search_results":reuslt})
print(response)

