import os
import sys
import time
import random
import numpy as np
import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import anthropic
from openai import AzureOpenAI
import openai

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

@dataclass
class _OpenAIMessage:
    content: str


@dataclass
class _OpenAIChoice:
    message: _OpenAIMessage


@dataclass
class _OpenAIChatCompletionResponse:
    """
    Minimal OpenAI-compatible response wrapper.
    This is intentionally tiny: `query_inference_server` only relies on `.choices[i].message.content`.
    """
    choices: list[_OpenAIChoice]
    raw: object | None = None


def _extract_anthropic_text(response: object) -> str:
    """
    Anthropic `messages.create(...)` returns an object with `.content` as a list of blocks.
    We concatenate any `.text` fields found; otherwise, fall back to string conversion.
    """
    try:
        blocks = getattr(response, "content", None)
        if isinstance(blocks, list):
            parts: list[str] = []
            for b in blocks:
                text = getattr(b, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
    except Exception:
        pass
    return str(response)


class _ClaudeChatCompletions:
    def __init__(self, client: anthropic.Anthropic):
        self._client = client

    def create(self, model: str, messages: list[dict], **kwargs):
        # OpenAI-style kwargs -> Anthropic-style kwargs
        max_tokens = kwargs.pop("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None:
            max_tokens = 1024

        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        stop = kwargs.pop("stop", None)

        system: str = ""
        anthropic_messages: list[dict] = []
        for m in messages or []:
            role = (m or {}).get("role", "user")
            content = (m or {}).get("content", "")

            # Anthropic uses `system=` parameter rather than a system role message.
            if role == "system":
                system = f"{system}\n{content}"
                continue

            if role not in ("user", "assistant"):
                role = "user"
            anthropic_messages.append({"role": role, "content": content})

        create_kwargs: dict = {}
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p
        if stop is not None:
            if isinstance(stop, str):
                stop = [stop]
            if isinstance(stop, list):
                create_kwargs["stop_sequences"] = stop

        response = self._client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            messages=anthropic_messages if anthropic_messages else [{"role": "user", "content": ""}],
            system=system,
            **create_kwargs,
        )
        text = _extract_anthropic_text(response)
        return _OpenAIChatCompletionResponse(
            choices=[_OpenAIChoice(message=_OpenAIMessage(content=text))],
            raw=response,
        )


class ClaudeOpenAICompatClient:
    """
    Provide an OpenAI-like interface:
      client.chat.completions.create(...)
    backed by Anthropic Claude.
    """
    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.chat = SimpleNamespace(completions=_ClaudeChatCompletions(self._client))


def create_inference_server(server_type):
    if server_type == "azure":
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT must be set when server_type='azure'."
            )
        return AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=os.environ.get("AZURE_API_KEY")
        )
    elif server_type == "openai":
        return openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1"
        )
    elif server_type in ("claude", "anthropic"):
        return ClaudeOpenAICompatClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported server type: {server_type}")

def query_inference_server(server, model_name: str, prompt: str, max_completion_tokens: int = 16384, retry_times: int = 5, full_response: bool = False, **kwargs):
    default_kwargs = {
        "temperature": 1.0,
        "max_completion_tokens": max_completion_tokens,
    }
    default_kwargs.update(kwargs)
    
    last_exception = None
    for attempt in range(retry_times):
        try:
            response = server.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                **default_kwargs
            )

            if full_response:
                return response
            
            # Check if response is a string (unexpected error case)
            # This shouldn't happen normally - OpenAI/AzureOpenAI should either return an object or raise an exception
            # But some proxies (like LiteLLM) might return strings in error cases
            if isinstance(response, str):
                print(f"Error: API returned string instead of response object (unexpected behavior)")
                print(f"This may indicate a proxy/middleware issue or API client bug")
                print(f"Response: {response}")
                raise ValueError(f"API returned string instead of response object: {response}")
            
            # Check if response has the expected structure
            if not hasattr(response, 'choices'):
                print(f"Error: Response object missing 'choices' attribute")
                print(f"Response type: {type(response)}")
                print(f"Response attributes: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
                print(f"Response: {response}")
                raise AttributeError(f"Response object missing 'choices' attribute. Type: {type(response)}")
            
            try:
                outputs = [choice.message.content for choice in response.choices]
            except Exception as e:
                print(f"Error: Failed to extract outputs from response: {e}")
                print(f"Response type: {type(response)}")
                print(f"Response: {response}")
                raise
            
            # Success - break out of retry loop
            break
            
        except Exception as e:
            last_exception = e
            # API call failed - this is the expected way for errors to be reported
            print(f"Error: API call failed with exception (attempt {attempt + 1}/{retry_times}): {e}")
            print(f"Exception type: {type(e)}")
            
            # If this is the last attempt, raise the exception
            if attempt == retry_times - 1:
                raise
            else:
                # Wait before retrying (exponential backoff with jitter)
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
    
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference server with a prompt")
    parser.add_argument("--server_type", type=str, default="azure", choices=["azure", "openai", "claude"])
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use")
    parser.add_argument("--prompt", type=str, default="Hello, world! Please respond with a brief greeting.", help="Prompt to send to the model")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--max_completion_tokens", type=int, default=16384, help="Maximum completion tokens")
    
    args = parser.parse_args()
    
    print("Creating inference server...")
    server = create_inference_server(args.server_type)
    print(f"Inference server created successfully!")
    
    print(f"\nSending prompt to model '{args.model_name}':")
    print(f"'{args.prompt}'")
    print("\n" + "="*80 + "\n")
    
    try:
        response = query_inference_server(
            server, 
            model_name=args.model_name,
            prompt=args.prompt,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature
        )
        print("Response:")
        print(response)
        print("\n" + "="*80)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)