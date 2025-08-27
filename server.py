import argparse
from mcp.server.fastmcp import FastMCP
import json
from typing import List, Dict, Optional

mcp = FastMCP("Insurance-Test")

# Load JSON data
with open("code_assurances.json", "r", encoding="utf-8") as f:
    code_data = json.load(f)

@mcp.tool()
def search_code_assurances(query: str) -> str:
    """Search the Code des Assurances for articles matching the query.

    Args:
        query (str): The search query, which can be an article ID, keyword, or phrase.

    Returns:
        str: JSON string containing a list of matching articles with their details.
    """
    results = []
    query_lower = query.lower()
    
    for article in code_data["articles"]:
        if article["article_id"].lower() == query.lower() or query_lower in article["article_id"].lower() or article["article_id"].lower() in query_lower:
            results.append({
                "article_id": article["article_id"],
                "content": article["content"],
                "hierarchy": article["hierarchy"],
                "references": article["references"],
                "referenced_by": article["referenced_by"],
            })
            break
    
    return json.dumps({"articles": results})

if __name__ == "__main__":
    print("Starting Code des Assurances server...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    args = parser.parse_args()
    mcp.run(args.server_type)