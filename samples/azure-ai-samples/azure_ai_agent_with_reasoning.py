"""
Agent Service v2 — Reasoning Effort & Reasoning Tokens Test
============================================================
Uses AzureAIClient (from agent-framework-azure-ai), which:
  • Creates versioned agents on the Foundry service via AIProjectClient
  • Supports reasoning options (effort, summary) via dict-based options (same as Responses API)
  • Exposes reasoning tokens via response usage_details

Prerequisites:
  pip install agent-framework-azure-ai azure-identity

Environment variables:
  AZURE_AI_PROJECT_ENDPOINT          – e.g. https://<account>.services.ai.azure.com/api/projects/<project>
  AZURE_AI_MODEL_DEPLOYMENT_NAME     – e.g. gpt-5-mini
"""

import asyncio
import os

from agent_framework.azure import AzureAIClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv(override=True)


async def test_reasoning(effort: str) -> None:
    """Test a specific reasoning effort level and inspect tokens."""

    print(f"\n{'='*60}")
    print(f"  Reasoning Effort: {effort.upper()}")
    print(f"{'='*60}")

    async with (
        AzureCliCredential() as credential,
        AzureAIClient(
            credential=credential,
            agent_name=f"reasoning-test-{effort}",
            model_deployment_name=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
        ) as client,
    ):

        response = await client.get_response(
            "What is 25 * 47 + 133? Walk me through the math step by step.",
            options={
                "reasoning": {"effort": effort, "summary": "concise"},
            },
        )

        # ── Print response text ──
        print(f"\nResponse:\n{response}\n")

        # ── Inspect usage & reasoning tokens ──
        usage = response.usage_details
        if usage:
            print(f"Token Usage:")
            print(f"  Input tokens:     {usage.get('input_token_count')}")
            print(f"  Output tokens:    {usage.get('output_token_count')}")
            print(f"  Total tokens:     {usage.get('total_token_count')}")

            # UsageDetails is now a TypedDict (plain dict) — reasoning tokens are just dict keys
            reasoning_tokens = usage.get("reasoning_tokens") or usage.get("openai.reasoning_tokens")
            cached_tokens = usage.get("cached_input_tokens") or usage.get("openai.cached_input_tokens")
            print(f"  Reasoning tokens: {reasoning_tokens}")
            print(f"  Cached input:     {cached_tokens}")

            output_tokens = usage.get('output_token_count')
            if output_tokens and reasoning_tokens:
                pct = (reasoning_tokens / output_tokens) * 100
                print(f"  Reasoning ratio:  {pct:.1f}% of output tokens")

            # Print all keys in usage dict
            print(f"  All usage keys: {dict(usage)}")
        else:
            print("  (No usage details available)")


async def test_with_tools(effort: str) -> None:
    """Test reasoning with function calling."""
    from typing import Annotated
    from pydantic import Field

    def calculate(expression: Annotated[str, Field(description="Math expression to evaluate")]) -> str:
        """Evaluate a math expression."""
        try:
            result = eval(expression)  # noqa: S307
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    print(f"\n{'='*60}")
    print(f"  Reasoning Effort: {effort.upper()} (with function calling)")
    print(f"{'='*60}")

    async with (
        AzureCliCredential() as credential,
        AzureAIClient(
            credential=credential,
            model_deployment_name=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
        ).as_agent(
            name=f"reasoning-tools-test-{effort}",
            instructions="You are a math assistant. Use the calculate tool when asked to compute something.",
            tools=calculate,
        ) as agent,
    ):

        response = await agent.run(
            "Calculate the factorial of 7, then divide by 42. Use the calculate tool.",
            options={
                "reasoning": {"effort": effort},
            },
        )

        print(f"\nResponse:\n{response}\n")

        usage = response.usage_details
        if usage:
            reasoning_tokens = usage.get("reasoning_tokens") or usage.get("openai.reasoning_tokens")
            print(f"Token Usage: input={usage.get('input_token_count')}, "
                  f"output={usage.get('output_token_count')}, "
                  f"reasoning={reasoning_tokens}")
            print(f"  All usage keys: {dict(usage)}")


async def main() -> None:
    print("Agent Service v2 — Reasoning Effort & Token Inspection")
    print("=" * 60)

    # ── Compare reasoning efforts ──
    for effort in ["low", "medium", "high"]:
        await test_reasoning(effort)

    # ── Test with function calling ──
    await test_with_tools("high")


if __name__ == "__main__":
    asyncio.run(main())