"""Model selection. Bedrock Nova Lite by default; NVIDIA-hosted fallback if USE_NVIDIA=1
and NVIDIA_API_KEY is set."""
import os
import re


def strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks (and stray tags) the model may print."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return re.sub(r"</?thinking>", "", text).strip()


def get_model():
    if os.environ.get("USE_NVIDIA") != "1" or not os.environ.get("NVIDIA_API_KEY"):
        from strands.models import BedrockModel

        return BedrockModel(
            model_id="us.amazon.nova-lite-v1:0",
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
        )

    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        client_args={
            "api_key": os.environ["NVIDIA_API_KEY"],
            "base_url": "https://integrate.api.nvidia.com/v1",
        },
        model_id=os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
    )
