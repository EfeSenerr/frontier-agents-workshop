# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from agent_framework import ChatAgent, CitationAnnotation
from agent_framework.azure import AzureAIAgentClient
from azure.ai.agents.aio import AgentsClient
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(override=True)
"""
Azure AI Agent with Azure AI Search Example - Multi-Turn Conversation

This sample demonstrates how to create an Azure AI agent that uses Azure AI Search
to search through indexed data and answer user questions in a multi-turn conversation.

Prerequisites:
1. Set AZURE_AI_PROJECT_ENDPOINT and AZURE_AI_MODEL_DEPLOYMENT_NAME environment variables
2. Ensure you have an Azure AI Search connection configured in your Azure AI project
3. The search index should exist in your Azure AI Search service

NOTE: To ensure consistent search tool usage:
- Include explicit instructions for the agent to use the search tool
- Mention the search requirement in your queries
- Use `tool_choice="required"` to force tool usage

Type 'quit', 'exit', or 'q' to end the conversation.
"""


async def main() -> None:
    """Main function demonstrating Azure AI agent with multi-turn conversation."""
    print("=== Azure AI Agent with Azure AI Search - Multi-Turn Conversation ===")
    print("Type 'quit', 'exit', or 'q' to end the conversation.\n")

    # Create the client and manually create an agent with Azure AI Search tool
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as project_client,
        AgentsClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential) as agents_client,
    ):
        ai_search_conn_id = os.environ["AI_SEARCH_PROJECT_CONNECTION_ID"]

        # 1. Create Azure AI agent with the search tool
        azure_ai_agent = await agents_client.create_agent(
            model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
            name="GrecoSearchAgent",
            instructions=(
                "You are a helpful agent that searches information using Azure AI Search. "
                "Always use the search tool and index to find data and provide accurate information."
            ),
            temperature=0.1,
            tools=[{"type": "azure_ai_search"}],
            tool_resources={
                "azure_ai_search": {
                    "indexes": [
                        {
                            "index_connection_id": ai_search_conn_id,
                            "index_name": os.environ["AI_SEARCH_INDEX_NAME"],
                            "query_type": "vector_semantic_hybrid",
                            "top_k": 20,
                        }
                    ]
                }
            },
        )

        # 2. Create chat client with the existing agent
        chat_client = AzureAIAgentClient(agents_client=agents_client, agent_id=azure_ai_agent.id)

        try:
            async with ChatAgent(
                chat_client=chat_client,
                instructions=("You are a helpful agent that uses the search tool and index to find information."),
            ) as agent:
                print("This agent uses Azure AI Search tool to search data.")
                print("The conversation maintains context across multiple turns.\n")

                # 3. Multi-turn conversation loop
                while True:
                    try:
                        user_input = input("User: ").strip()
                    except EOFError:
                        break

                    # Check for exit commands
                    if user_input.lower() in ['quit', 'exit', 'q', '']:
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("\nEnding conversation. Goodbye!")
                        break

                    print("Agent: ", end="", flush=True)

                    # Stream the response and collect citations
                    citations: list[CitationAnnotation] = []
                    async for chunk in agent.run_stream(user_input, tool_choice="required"):
                        if chunk.text:
                            print(chunk.text, end="", flush=True)

                        # Collect citations from Azure AI Search responses
                        for content in getattr(chunk, "contents", []):
                            annotations = getattr(content, "annotations", [])
                            if annotations:
                                citations.extend(annotations)

                    print()

                    # Display collected citations
                    if citations:
                        print("\nCitations:")
                        for i, citation in enumerate(citations, 1):
                            print(f"  [{i}] {citation.url}")

                    print()  # Add spacing between turns

                print("\n" + "=" * 50)
                print("Multi-turn conversation completed!")

        finally:
            # Clean up the agent manually
            await agents_client.delete_agent(azure_ai_agent.id)


if __name__ == "__main__":
    asyncio.run(main())
