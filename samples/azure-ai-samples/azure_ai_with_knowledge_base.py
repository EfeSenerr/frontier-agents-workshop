# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os

from agent_framework.azure import AzureAIAgentClient
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import MCPTool
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(override=True)

"""
Azure AI Agent with Agentic Retrieval (Knowledge Base) Example

This sample demonstrates how to use agentic retrieval via knowledge bases in Azure AI Search.
Agentic retrieval provides advanced RAG capabilities:
- LLM-based query planning and decomposition
- Parallel subquery execution
- Automatic semantic reranking
- Structured responses with citations

Prerequisites:
1. Create a knowledge base in Azure AI Search with your indexed data
2. Create a project connection in Foundry pointing to the knowledge base MCP endpoint
3. Set environment variables:
   - AZURE_AI_PROJECT_ENDPOINT
   - AZURE_AI_MODEL_DEPLOYMENT_NAME
   - KNOWLEDGE_BASE_MCP_ENDPOINT (format: {search_endpoint}/knowledgebases/{kb_name}/mcp?api-version=2025-11-01-preview)
   - KB_PROJECT_CONNECTION_NAME (your project connection name)
"""


async def main() -> None:
    async with (
        AzureCliCredential() as credential,
        AIProjectClient(
            endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential
        ) as project_client,
    ):
        # Create MCP tool for knowledge base access
        mcp_kb_tool = MCPTool(
            server_label="knowledge-base",
            server_url=os.environ["KNOWLEDGE_BASE_MCP_ENDPOINT"],
            require_approval="never",
            allowed_tools=["knowledge_base_retrieve"],
            project_connection_id=os.environ["KB_PROJECT_CONNECTION_NAME"],
        )

        # Create agent with knowledge base tool
        azure_ai_agent = await project_client.agents.create_version(
            agent_name="KnowledgeBaseAgent",
            definition={
                "kind": "prompt",
                "model": os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
                "instructions": """You are a helpful assistant that uses a knowledge base to answer questions.
You must always use the knowledge base tool to retrieve information.
Always provide citations using the tool and render them as: `[message_idx:search_idx†source_name]`.""",
                "tools": [mcp_kb_tool],
            },
            description="Agent using agentic retrieval for knowledge base access",
        )

        print(f"Created agent: {azure_ai_agent.name} (version {azure_ai_agent.version})")

        # Create chat client for the agent
        chat_client = AzureAIAgentClient(
            project_client=project_client,
            async_credential=credential,
            agent_name=azure_ai_agent.name,
            agent_version=azure_ai_agent.version,
        )

        try:
            # Create agent instance for interaction
            async with chat_client.create_agent(
                name=azure_ai_agent.name,
                instructions=azure_ai_agent.definition.get("instructions"),
            ) as agent:
                # Interactive loop
                print("\n" + "=" * 60)
                print("Chat with Knowledge Base Agent (type 'quit' to exit)")
                print("=" * 60 + "\n")

                while True:
                    user_input = input("You: ").strip()

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("Goodbye!")
                        break

                    if not user_input:
                        continue

                    print(f"\nAgent: ", end="", flush=True)
                    result = await agent.run(user_input)
                    print(result)
                    print()

        finally:
            # Clean up - delete the agent version
            # await project_client.agents.delete_version(
            #     agent_name=azure_ai_agent.name, agent_version=azure_ai_agent.version
            # )
            print(f"\nDeleted agent version: {azure_ai_agent.name} v{azure_ai_agent.version}")


if __name__ == "__main__":
    asyncio.run(main())
