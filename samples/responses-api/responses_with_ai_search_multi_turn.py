"""
Azure OpenAI Responses API with Azure AI Search - Multi-Turn Conversation
Uses DefaultAzureCredential (az login) for secure production authentication.
No API keys required - just run 'az login' before executing.

Required Environment Variables:
- AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
- AZURE_MODEL_NAME: Your model deployment name (e.g., gpt-5-mini)
- AI_SEARCH_ENDPOINT: Your Azure AI Search endpoint
- AI_SEARCH_INDEX_NAME: Your search index name

Optional:
- AI_SEARCH_API_KEY: If set, uses API key auth for AI Search (otherwise Entra ID)
- AI_SEARCH_SEMANTIC_CONFIG: Semantic configuration name (default: "default")
- AI_SEARCH_VECTOR_FIELD: Vector field name (default: "text_vector")

Type 'quit', 'exit', or 'q' to end the conversation.
"""

import os
import json
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv(override=True)

# Setup DefaultAzureCredential (uses az login, managed identity, etc.)
credential = DefaultAzureCredential()

# Setup Azure OpenAI client with DefaultAzureCredential
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

client = OpenAI(
    base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}openai/v1/",
    api_key=token_provider(),
)

# Azure AI Search client - prefer DefaultAzureCredential, fallback to API key if set
ai_search_api_key = os.environ.get("AI_SEARCH_API_KEY", "").strip()
if ai_search_api_key:
    print("AI Search: Using API Key authentication")
    search_credential = AzureKeyCredential(ai_search_api_key)
else:
    print("AI Search: Using DefaultAzureCredential (requires 'Search Index Data Reader' role)")
    search_credential = credential

search_client = SearchClient(
    endpoint=os.environ["AI_SEARCH_ENDPOINT"],
    index_name=os.environ["AI_SEARCH_INDEX_NAME"],
    credential=search_credential
)

print("=" * 60)
print("🔐 Authentication: Using DefaultAzureCredential (az login)")
print(f"📍 Azure OpenAI Endpoint: {os.environ['AZURE_OPENAI_ENDPOINT']}")
print(f"🔍 AI Search Endpoint: {os.environ['AI_SEARCH_ENDPOINT']}")
print(f"📚 AI Search Index: {os.environ['AI_SEARCH_INDEX_NAME']}")
print("=" * 60)


def search_documents(query: str, top: int = 5) -> str:
    """Search Azure AI Search and return results as JSON string."""
    from azure.search.documents.models import VectorizableTextQuery
    
    print(f"\n  🔍 Searching: {query}")
    
    # Check if query looks like an exact document name pattern
    query_words = query.strip().split()
    is_document_name = (
        len(query_words) <= 2 and
        "_" in query and
        not any(word in query.lower() for word in ["content", "article", "about", "what", "how"])
    )
    
    try:
        if is_document_name:
            # Search using the document name - fetch more results then filter by title
            doc_name = query_words[0].replace(".pdf", "")
            print(f"  📄 Searching for document: {doc_name}")
            # Get more results since we'll filter by title afterwards
            all_results = list(search_client.search(
                search_text="*",  # Get all documents
                top=50,  # Fetch more to find matching titles
                select=["chunk", "title"]
            ))
            # Filter results to only include matching titles
            results = [r for r in all_results if doc_name.lower() in r.get("title", "").lower()][:top]
            print(f"  📄 Filtered to {len(results)} chunks from '{doc_name}'")
        else:
            vector_field = os.getenv("AI_SEARCH_VECTOR_FIELD", "text_vector")
            semantic_config = os.getenv("AI_SEARCH_SEMANTIC_CONFIG", "default")
            
            print(f"  🧠 Hybrid search (semantic + vector) with field: {vector_field}")
            results = list(search_client.search(
                search_text=query,
                top=top,
                query_type="semantic",
                semantic_configuration_name=semantic_config,
                vector_queries=[
                    VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=top,
                        fields=vector_field,
                    )
                ],
                select=["chunk", "title"]
            ))
    except Exception as e:
        print(f"  ⚠️ Search failed ({e}), using simple search...")
        results = list(search_client.search(
            search_text=query,
            top=top,
            select=["chunk", "title"]
        ))
    
    print(f"  📥 Found {len(results)} results")
    
    # Format results for the model
    formatted = []
    for doc in results:
        title = doc.get("title", "N/A")
        content = doc.get("chunk", "")[:500]
        formatted.append({"title": title, "content": content})
    
    return json.dumps(formatted, indent=2)


# Define the search tool
search_tool = {
    "type": "function",
    "name": "search_knowledge_base",
    "description": "Search the knowledge base for documents. For finding a specific document by name, use just the document name (e.g., 'document.pdf'). For content search, use descriptive keywords. Call this tool separately for each document you need to find.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Either a document filename (e.g., 'document.pdf') or descriptive search terms"
            }
        },
        "required": ["query"]
    }
}

SYSTEM_PROMPT = """You are a helpful assistant that searches a knowledge base to answer questions. 
When comparing documents, search for each document separately by its exact filename. 
Make one search call per document. Always use the search tool to find information before answering."""


def process_tool_calls(response, previous_response_id: str) -> tuple:
    """Process tool calls and return the final response with text output."""
    max_iterations = 5
    current_response = response
    
    for iteration in range(max_iterations):
        tool_outputs = []
        
        for output in current_response.output:
            if output.type == "function_call":
                if output.name == "search_knowledge_base":
                    args = json.loads(output.arguments)
                    search_results = search_documents(args["query"])
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": search_results
                    })
        
        if not tool_outputs:
            break
        
        should_allow_tools = iteration < max_iterations - 2
        
        current_response = client.responses.create(
            model=os.environ.get("AZURE_MODEL_NAME", "gpt-5-mini"),
            previous_response_id=current_response.id,
            input=tool_outputs,
            tools=[search_tool],
            tool_choice="auto" if should_allow_tools else "none",
        )
    
    # Extract text response
    text_output = ""
    for output in current_response.output:
        if hasattr(output, 'content') and output.content is not None:
            for content in output.content:
                if hasattr(content, 'text'):
                    text_output += content.text
    
    return current_response, text_output


def main():
    """Multi-turn conversation with Azure AI Search."""
    print("\n" + "=" * 60)
    print("🤖 Azure OpenAI Responses API with AI Search")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'q' to end the conversation.\n")
    
    # Track conversation using previous_response_id
    previous_response_id = None
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q', '']:
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
            break
        
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # Build input - include system message on first turn
            input_messages = []
            if previous_response_id is None:
                input_messages.append({"role": "system", "content": SYSTEM_PROMPT})
            input_messages.append({"role": "user", "content": user_input})
            
            # Create response (with or without conversation context)
            create_kwargs = {
                "model": os.environ.get("AZURE_MODEL_NAME", "gpt-5-mini"),
                "tools": [search_tool],
                "tool_choice": "auto",
                "input": input_messages,
            }
            
            if previous_response_id:
                create_kwargs["previous_response_id"] = previous_response_id
            
            response = client.responses.create(**create_kwargs)
            
            # Check if there are tool calls to process
            has_tool_calls = any(
                output.type == "function_call" 
                for output in response.output
            )
            
            if has_tool_calls:
                print("(searching...)\n")
                final_response, text_output = process_tool_calls(response, response.id)
                previous_response_id = final_response.id
            else:
                # Extract text directly
                text_output = ""
                for output in response.output:
                    if hasattr(output, 'content') and output.content is not None:
                        for content in output.content:
                            if hasattr(content, 'text'):
                                text_output += content.text
                previous_response_id = response.id
            
            print(text_output if text_output else "(No response)")
            print()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print()


if __name__ == "__main__":
    main()
