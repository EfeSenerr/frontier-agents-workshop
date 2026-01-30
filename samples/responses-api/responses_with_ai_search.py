"""
Azure OpenAI Responses API with Azure AI Search Tool Call
Uses DefaultAzureCredential (az login) for secure production authentication.
No API keys required - just run 'az login' before executing.

Note: For AI Search, you need "Search Index Data Reader" role assigned to your identity.
      If RBAC is not configured, set AI_SEARCH_API_KEY in .env as fallback.
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
    api_key=token_provider(),  # Call the token provider to get the actual token
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
    
    # Log AI Search request
    print("\n" + "=" * 60)
    print("📡 AZURE AI SEARCH - REQUEST")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Top results requested: {top}")
    print(f"Index: {os.environ['AI_SEARCH_INDEX_NAME']}")
    
    # Check if query looks like an exact document name pattern (e.g., "insurance_times_bank" or "insurance_times_bank.pdf")
    # Only trigger if the query is primarily a document name (short, with underscores, no spaces or few words)
    query_words = query.strip().split()
    is_document_name = (
        len(query_words) <= 2 and  # Short query (1-2 words)
        "_" in query and  # Contains underscores
        not any(word in query.lower() for word in ["content", "article", "about", "what", "how"])
    )
    
    try:
        if is_document_name:
            # Search using the document name - fetch more results then filter by title
            doc_name = query_words[0].replace(".pdf", "")  # Remove .pdf if present
            print(f"Detected document name pattern: {doc_name}")
            print(f"Searching for document: {doc_name}")
            # Get more results since we'll filter by title afterwards
            all_results = list(search_client.search(
                search_text="*",  # Get all documents
                top=50,  # Fetch more to find matching titles
                select=["chunk", "title"]
            ))
            # Filter results to only include matching titles
            results = [r for r in all_results if doc_name.lower() in r.get("title", "").lower()][:top]
            print(f"Filtered to {len(results)} chunks from '{doc_name}'")
        else:
            # Content-based search using vector + semantic hybrid
            vector_field = os.getenv("AI_SEARCH_VECTOR_FIELD", "text_vector")
            semantic_config = os.getenv("AI_SEARCH_SEMANTIC_CONFIG", "default")
            
            print(f"Using semantic search with vector field: {vector_field}")
            results = list(search_client.search(
                search_text=query,
                top=top,
                query_type="vector_semantic_hybrid",
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
        print(f"Search failed ({e}), falling back to simple full-text search...")
        results = list(search_client.search(
            search_text=query,
            top=top,
            select=["chunk", "title"]
        ))
    
    # Log AI Search response
    print("\n" + "-" * 60)
    print("📥 AZURE AI SEARCH - RESPONSE")
    print("-" * 60)
    print(f"Documents returned: {len(results)}")
    
    # Format results for the model
    formatted = []
    for i, doc in enumerate(results, 1):
        title = doc.get("title", "N/A")
        content = doc.get("chunk", "")[:500]  # Limit content length
        score = doc.get("@search.score", "N/A")
        
        print(f"\n  [{i}] Title: {title}")
        print(f"      Score: {score}")
        print(f"      Content preview: {content[:100]}...")
        
        formatted.append({
            "title": title,
            "content": content
        })
    
    print("\n" + "=" * 60)
    
    return json.dumps(formatted, indent=2)


# Define the search tool for the Responses API
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

# Get user query
print("\n" + "=" * 60)
print("🚀 AZURE OPENAI RESPONSES API WITH AI SEARCH")
print("=" * 60)
print(f"Model: {os.environ.get('AZURE_MODEL_NAME', 'gpt-5-mini')}")
print("\nEnter your query (e.g., 'compare document1.pdf and document2.pdf'):")
print("Run debug_index.py to see available documents in your index.\n")

user_query = input("You: ").strip()
if not user_query:
    print("No query provided. Exiting.")
    exit(0)

print(f"\nProcessing query: {user_query}")

response = client.responses.create(
    model=os.environ.get("AZURE_MODEL_NAME", "gpt-5-mini"),
    tools=[search_tool],
    tool_choice="required",  # Force the model to use the tool
    input=[
        {"role": "system", "content": "You are a helpful assistant that searches a knowledge base. When comparing documents, search for each document separately by its exact filename. Make one search call per document."},
        {"role": "user", "content": user_query}
    ],
)

print("\nInitial Response:")
print(response.model_dump_json(indent=2))

# Handle function calls in a loop (model may need multiple rounds)
max_iterations = 5
current_response = response

for iteration in range(max_iterations):
    tool_outputs = []
    
    for output in current_response.output:
        if output.type == "function_call":
            print(f"\n{'=' * 60}")
            print(f"🔧 TOOL CALL DETECTED (iteration {iteration + 1})")
            print(f"{'=' * 60}")
            print(f"Function: {output.name}")
            print(f"Arguments: {output.arguments}")
            print(f"Call ID: {output.call_id}")
            
            if output.name == "search_knowledge_base":
                args = json.loads(output.arguments)
                search_results = search_documents(args["query"])
                
                print(f"\n{'=' * 60}")
                print("📦 SEARCH RESULTS (JSON for model)")
                print("=" * 60)
                print(search_results)
                
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": search_results
                })
            else:
                raise ValueError(f"Unknown function call: {output.name}")
    
    # If no tool calls, we're done
    if not tool_outputs:
        break
    
    print(f"\n{'=' * 60}")
    print(f"Sending {len(tool_outputs)} function result(s) back to the model...")
    print("=" * 60)
    
    # Determine if we should allow more tool calls or force a final answer
    # Allow more calls in early iterations, force text in later ones
    should_allow_tools = iteration < max_iterations - 2
    
    current_response = client.responses.create(
        model=os.environ.get("AZURE_MODEL_NAME", "gpt-5-mini"),
        previous_response_id=current_response.id,
        input=tool_outputs,
        tools=[search_tool],
        tool_choice="auto" if should_allow_tools else "none",
    )
    
    print(f"\nResponse (iteration {iteration + 1}):")
    print(current_response.model_dump_json(indent=2))

# Extract and print the final text response
print(f"\n{'=' * 60}")
print("🤖 MODEL'S FINAL ANSWER")
print("=" * 60)

for output in current_response.output:
    if hasattr(output, 'content') and output.content is not None:
        for content in output.content:
            if hasattr(content, 'text'):
                print(content.text)
