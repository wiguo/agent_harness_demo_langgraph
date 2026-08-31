"""The tool list the graph runs with. Adding a tool = one import + one line."""

from .query_cases import query_cases_tool
from .search_documents import search_documents_tool

TOOLS = [search_documents_tool, query_cases_tool]
