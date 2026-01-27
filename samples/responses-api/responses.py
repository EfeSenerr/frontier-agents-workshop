"""
Classic RAG Pattern: Azure AI Search + GPT-5 via Responses API

This script demonstrates how to use GPT-5 models with Azure AI Search
by querying the search index directly and passing results to the Responses API.

This approach works with GPT-5 models, unlike the Agent Service azure_ai_search tool
which only supports gpt-4o/gpt-4.1 models.

Prerequisites:
1. Fill in AI_SEARCH_ENDPOINT and AI_SEARCH_INDEX_NAME in .env
2. pip install openai azure-search-documents azure-identity python-dotenv
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv(override=True)


def get_search_client() -> SearchClient:
    """Create Azure AI Search client."""
    endpoint = os.environ["AI_SEARCH_ENDPOINT"]
    index_name = os.environ["AI_SEARCH_INDEX_NAME"]
    api_key = os.getenv("AI_SEARCH_API_KEY", "")

    if api_key:
        credential = AzureKeyCredential(api_key)
    else:
        credential = DefaultAzureCredential()

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential
    )


def search_documents(query: str, top_k: int = 10) -> list[dict]:
    """
    Search Azure AI Search index and return results.
    
    Uses vector semantic hybrid search when AI_SEARCH_SEMANTIC_CONFIG and 
    AI_SEARCH_VECTOR_FIELD are set in .env.
    """
    from azure.search.documents.models import VectorizableTextQuery
    
    search_client = get_search_client()
    semantic_config = os.getenv("AI_SEARCH_SEMANTIC_CONFIG", "")
    vector_field = os.getenv("AI_SEARCH_VECTOR_FIELD", "")
    
    search_kwargs = {
        "search_text": query,
        "top": top_k,
    }
    
    if semantic_config:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = semantic_config
    
    if vector_field:
        search_kwargs["vector_queries"] = [
            VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=top_k,
                fields=vector_field,
            )
        ]
    
    # Log search type
    if semantic_config and vector_field:
        print(f"   Query type: vector_semantic_hybrid")
    elif semantic_config:
        print(f"   Query type: semantic")
    elif vector_field:
        print(f"   Query type: vector")
    else:
        print(f"   Query type: simple full-text")
    
    results = search_client.search(**search_kwargs)
    return [dict(result) for result in results]


def format_search_results(documents: list[dict]) -> str:
    """Format search results into context string for the LLM."""
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        # Common field names - adjust based on your index schema
        title = doc.get("title") or doc.get("name") or doc.get("fileName") or f"Document {i}"
        content = doc.get("content") or doc.get("chunk") or doc.get("text") or str(doc)
        
        # Truncate long content
        if len(content) > 2000:
            content = content[:2000] + "..."
        
        context_parts.append(f"[{i}] {title}\n{content}")
    
    return "\n\n---\n\n".join(context_parts)


def ask_with_search(query: str, model: str = None) -> str:
    """
    Query Azure AI Search, then use GPT to generate an answer.
    
    This is the classic RAG pattern that works with all models including GPT-5.
    """
    # 1. Search for relevant documents
    print(f"🔍 Searching for: {query}")
    documents = search_documents(query, top_k=10)
    print(f"📄 Found {len(documents)} documents")
    
    # 2. Format context
    context = format_search_results(documents)
    
    # 3. Create OpenAI client for Responses API (using Entra ID auth)
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version="2025-03-01-preview",
    )
    
    # 4. Use the model to generate an answer
    model_name = model or os.getenv("AZURE_MODEL_NAME", "gpt-5-mini")
    
    print(f"🤖 Generating answer using {model_name}...")
    
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on the provided context. "
                    "Always cite your sources using [number] references. "
                    "If the context doesn't contain enough information, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context from search:\n\n{context}\n\n---\n\nQuestion: {query}"
            }
        ]
    )
    
    return response.output_text


def main():
    """Main function to demonstrate Azure AI Search with Responses API."""
    print("=" * 60)
    print("Azure AI Search + GPT-5 via Responses API")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = ["AZURE_OPENAI_ENDPOINT", 
                     "AI_SEARCH_ENDPOINT", "AI_SEARCH_INDEX_NAME"]
    
    missing = [var for var in required_vars if not os.getenv(var) or "your-" in os.getenv(var, "")]
    if missing:
        print(f"\n❌ Please set these environment variables in .env:")
        for var in missing:
            print(f"   - {var}")
        return
    
    # Example query - modify as needed
    query = input("\nEnter your search query (or press Enter for default): ").strip()
    if not query:
        query = "What are the main topics covered in the documents?"
    
    try:
        answer = ask_with_search(query)
        print("\n" + "=" * 60)
        print("📝 Answer:")
        print("=" * 60)
        print(answer)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
