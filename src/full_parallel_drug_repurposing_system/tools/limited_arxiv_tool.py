from crewai_tools import ArxivPaperTool
from typing import Type, Any
from pydantic import BaseModel, Field

class CustomArxivTool(ArxivPaperTool):
    name: str = "Arxiv Search"
    description: str = "Search Arxiv for scientific papers. Returns summaries of relevant papers."

    def _run(self, query: str) -> str:
        return super()._run(query)


from crewai.tools import BaseTool
import arxiv 

class LimitedArxivTool(BaseTool):
    name: str = "Arxiv Search"
    description: str = "Search for scientific papers on Arxiv. Input: query string."

    def _run(self, query: str) -> str:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=8,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for r in client.results(search):
            results.append(f"Title: {r.title}\nSummary: {r.summary}\nURL: {r.pdf_url}\n")
        return "\n\n".join(results)
