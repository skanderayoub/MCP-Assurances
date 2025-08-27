from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import nest_asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, ToolCall
from llama_index.core.workflow import Context
import asyncio

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
llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm

# Initialize client and build agent
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

async def get_tools():
    return await mcp_tool.to_tool_list_async()

async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Our Database software.",
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

async def main():
    agent = await get_agent(mcp_tool)
    # create the agent context
    agent_context = Context(agent)

    while True:
        user_input = input("Enter your message: ")
        if user_input == "exit":
            break
        print("User: ", user_input)
        response = await handle_user_message(user_input, agent, agent_context, verbose=True)
        print("Agent: ", response)

if __name__ == "__main__":
    asyncio.run(main())