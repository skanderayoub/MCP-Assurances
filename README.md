# Code des Assurances NPC Assistant

This project provides a simple Python-based application to assist insurance workers in querying the French *Code des Assurances*. It consists of a server (`server.py`) that exposes a search tool via the MCP framework and a client (`client.py`) with a graphical user interface (GUI) built using `tkinter`. The application allows users to search for articles by article ID, keywords, or phrases, displaying structured results (article ID, summary, keywords, and truncated content).

## Requirements

- **Python 3.8+**
- Required Python packages:
  - `llama-index`
  - `python-dotenv`
  - `tkinter` (usually included with Python)
  - `mcp` (for `FastMCP` and `BasicMCPClient`)
- A `code_assurances.json` file containing the *Code des Assurances* data.
- An OpenAI API key stored in a `.env` file (e.g., `OPENAI_API_KEY=your-key-here`).
- Alternatively, a LLM on you local machine.

Install the required packages using:
```bash
pip install llama-index python-dotenv llama-index-llms-openai llama-index-llms-ollama "mcp[cli]"
```

## Setup

1. **Prepare the JSON Data**:
   - Run the `preprocess_file.py` to generate the `code_assurances.json`, ensure `LEGITEXT000006073984.pdf` is in the same directory.
   - Place the `code_assurances.json` file in the same directory as `server.py` and `client.py`. This file contains the *Code des Assurances* articles to be queried.
   - Ensure the JSON file is correctly formatted with fields like `article_id`, `content`, `summary`, and `keywords`.

2. **Set Up Environment**:
   - Create a `.env` file in the project directory with your OpenAI API key:
     ```env
     OPENAI_API_KEY=your-key-here
     ```

3. **File Structure**:
   - `server.py`: Runs the server, exposing a search tool via the MCP framework.
   - `client.py`: Runs the GUI client, allowing users to query articles.
   - `code_assurances.json`: Contains the *Code des Assurances* data.
   - `.env`: Stores the OpenAI API key.

## Running the Application

1. **Start the Server**:
   - Open a terminal in the project directory.
   - Run the server using:
     ```bash
     uv run server.py --server_type=sse
     ```
   - The server will start and load the `code_assurances.json` file, exposing the `search_code_assurances` tool via the MCP framework.
   - You should see: `Starting Code des Assurances server...`.

2. **Start the Client**:
   - Open another terminal in the project directory.
   - Run the client using:
     ```bash
     python client.py
     ```
   - A GUI window will open with an entry field, a "Search" button, a scrollable results area, and an "Exit" button.

3. **Using the GUI**:
   - **Enter a Query**: Type a query in the entry field (e.g., Resume the content of `Article L432-1`).
   - **Search**: Click the "Search" button or press Enter to submit the query.
   - **View Results**: The answer of the LLM will be displayed in the bottom view of the GUI.
   - **Exit**: Click the "Exit" button to close the application.

## Queries

- For now, the tools only queries articles by id. It can not find articles by content or keywords.

## How It Works

- **Server (`server.py`)**:
  - Loads `code_assurances.json` and exposes a `search_code_assurances` tool using the `@mcp.tool()` decorator.
  - The tool searches articles by article ID, keywords, or content and returns results as a JSON string.
  - Runs using the MCP framework with Server-Sent Events (SSE) for communication.

- **Client (`client.py`)**:
  - Provides a `tkinter`-based GUI for entering queries and displaying results.
  - Uses `BasicMCPClient` to connect to the server and call the `search_code_assurances` tool.
  - Parses JSON responses and displays structured article information in the GUI.

## Notes

- The GUI is minimal and designed for simplicity. Future enhancements could include query history, advanced filters, or export options.
- The application uses the OpenAI `gpt-4o-mini` model for natural language processing, which requires a valid API key but you can still you an local LLM using Ollama.
