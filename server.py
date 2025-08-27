import argparse
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Insurance-Test")

if __name__ == "__main__":
    print("Starting server...")
    
    # Debug Mode
    #  uv run mcp dev server.py

    # Production Mode
    # uv run server.py --server_type=sse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_type", type=str, default="sse", choices=["sse", "stdio"]
    )
    
    args = parser.parse_args()
    mcp.run(args.server_type)