"""
Test script: Agent Framework v2 with reasoning_effort and reasoning token inspection.

Uses AzureOpenAIResponsesClient (which wraps OpenAI Responses API)
to set reasoning effort on GPT-5-mini and inspect output_tokens_details.reasoning_tokens.

Prerequisites:
  pip install agent-framework --pre
  az login

Environment variables:
  AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
  AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME=gpt-5-mini  # your deployment name
"""

import asyncio
import os
from agent_framework import ChatAgent, ChatMessage
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(override=True)

async def main():
    # --- 1. Create the Responses Client targeting gpt-5-mini ---
    client = AzureOpenAIResponsesClient(
        credential=AzureCliCredential(),
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.environ.get("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME", "gpt-5-mini"),
        api_version="preview",
    )

    # --- 2. Call get_response with reasoning options ---
    # The "reasoning" key maps to the OpenAI Responses API ReasoningOptions TypedDict:
    #   effort: Literal["low", "medium", "high"]
    #   summary: Literal["auto", "concise", "detailed"]  (optional)
    message = "What are the three largest prime numbers below 1000? Explain your reasoning step by step."

    print(f"User: {message}\n")
    print("--- Calling with reasoning.effort = 'high' ---\n")

    response = await client.get_response(
        message,
        options={
            "reasoning": {
                "effort": "high",       # <-- This is how you set reasoning_effort in v2
                # "summary": "concise", # Optional: get a summary of reasoning
            },
        },
    )

    # --- 3. Print the response text ---
    print(f"Assistant: {response.text}\n")

    # --- 4. Inspect usage details including reasoning tokens ---
    usage = response.usage_details
    print("=" * 60)
    print("TOKEN USAGE DETAILS")
    print("=" * 60)

    if usage:
        print(f"  input_token_count:       {usage.input_token_count}")
        print(f"  output_token_count:      {usage.output_token_count}")
        print(f"  total_token_count:       {usage.total_token_count}")
        print()

        # The key field the customer is asking about:
        # Use .get() to avoid KeyError if key doesn't exist
        reasoning_tokens = usage.additional_counts.get("reasoning_tokens") or usage.additional_counts.get("openai.reasoning_tokens")
        # cached_input = usage.openai.cached_input_tokens

        print(f"  reasoning_tokens: {reasoning_tokens}")
        # print(f"  openai.cached_input_tokens: {cached_input}")
        print()

        if reasoning_tokens is not None:
            print(f"  >> Reasoning tokens ARE visible: {reasoning_tokens} tokens used for internal reasoning")
        else:
            print("  >> WARNING: reasoning_tokens is None - check model/deployment support")

        # Print ALL keys in additional_counts to see everything available
        print()
        print("  All additional_counts keys:")
        for key, value in usage.additional_counts.items():
            print(f"    {key}: {value}")
    else:
        print("  No usage details returned!")

    print()
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())