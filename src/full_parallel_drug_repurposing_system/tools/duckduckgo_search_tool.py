from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Any
from ddgs import DDGS
import json

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="The search query to find information on the internet.")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "A search engine. Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error executing DuckDuckGo search: {str(e)}"
