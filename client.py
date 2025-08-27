from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import nest_asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
import asyncio
import json
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

load_dotenv()
nest_asyncio.apply()

SYSTEM_PROMPT = """\
You are an AI assistant specialized in analyzing and extracting information from the French 'Code des Assurances.' Your primary task is to parse legal text, identify key insurance-related concepts, and provide accurate, concise, and contextually relevant responses. 

Follow these guidelines:
1. **Understand Legal Context**: Interpret the input in the context of French insurance law, focusing on terms like 'assurance,' 'contrat,' 'sinistre,' 'indemnisation,' 'responsabilité,' and other domain-specific terminology.
2. **Extract Key Information**: When processing text, prioritize extracting relevant keywords, provisions, or concepts using tools to interact with the provided file or database.
3. **Handle French Language**: Account for French linguistic nuances, including proper nouns, legal jargon, and multi-word expressions (e.g., 'responsabilité civile,' 'assurance maladie').
4. **Provide Structured Responses**: Summarize findings clearly, citing specific articles or sections when relevant (e.g., 'Article L121-1'). If clarification is needed, ask the user for additional context.
5. **Use Tools Effectively**: Leverage available tools to query or process the 'Code des Assurances' database, ensuring responses are grounded in the source text.

Respond in a professional and precise manner, avoiding irrelevant details. If the input is ambiguous, request clarification to ensure accuracy.
"""

# Setup LLM
from llama_index.core import Settings
#llm = OpenAI(model="gpt-4o-mini")
llm = Ollama(model="llama3.2:1b")
Settings.llm = llm

# Initialize MCP client
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

async def get_tools():
    return await mcp_tool.to_tool_list_async()

async def get_agent(tools):
    agent = FunctionAgent(
        name="InsuranceCodeAgent",
        description="An agent that assists insurance workers by parsing the French Code des Assurances.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    handler = agent.run(message_content, ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and type(event) == ToolCall:
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler
    return str(response)

class InsuranceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Code des Assurances NPC Assistant")
        self.root.geometry("600x400")

        # Initialize agent and context
        self.loop = asyncio.get_event_loop()
        self.tools = self.loop.run_until_complete(get_tools())
        self.agent = self.loop.run_until_complete(get_agent(self.tools))
        self.agent_context = Context(self.agent)

        # Create UI elements
        self.label = ttk.Label(root, text="Enter query (e.g., 'Article L432-1' or 'state guarantees'):")
        self.label.pack(pady=5)

        self.query_entry = ttk.Entry(root, width=50)
        self.query_entry.pack(pady=5)
        self.query_entry.bind("<Return>", self.search)  # Bind Enter key to search

        self.search_button = ttk.Button(root, text="Search", command=self.search)
        self.search_button.pack(pady=5)

        self.result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15)
        self.result_text.pack(pady=10, padx=10)

        self.exit_button = ttk.Button(root, text="Exit", command=self.exit)
        self.exit_button.pack(pady=5)

    def search(self, event=None):
        query = self.query_entry.get().strip()
        if not query:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please enter a query.\n")
            return

        # Run async search in the event loop
        response = self.loop.run_until_complete(
            handle_user_message(query, self.agent, self.agent_context, verbose=True)
        )

        # Clear previous results
        self.result_text.delete(1.0, tk.END)

        try:
            # Parse response as JSON
            parsed_response = json.loads(response)
            if not parsed_response.get("articles"):
                self.result_text.insert(tk.END, "No articles found matching the query.\n")
            else:
                self.result_text.insert(tk.END, "Search Results:\n\n")
                for article in parsed_response["articles"]:
                    self.result_text.insert(tk.END, f"Article {article['article_id']}:\n")
                    self.result_text.insert(tk.END, f"  Summary: {article['summary']}\n")
                    self.result_text.insert(tk.END, f"  Keywords: {', '.join(article['keywords'])}\n")
                    self.result_text.insert(tk.END, f"  Content: {article['content'][:200]}...\n\n")
        except json.JSONDecodeError:
            # Fallback to raw response
            self.result_text.insert(tk.END, f"Agent: {response}\n")

    def exit(self):
        self.root.quit()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = InsuranceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()