"""BEFORE: a recipe agent that only knows what you type. No tools."""
from strands import Agent

from model import get_model

model = get_model()

agent = Agent(
    model=model,
    system_prompt=(
        "You are Fridge Chef, a friendly recipe assistant. "
        "Suggest simple recipes based on the ingredients the user lists."
    ),
)

if __name__ == "__main__":
    import sys

    # Record this failure for your demo video (the "before").
    photo = (sys.argv[1] if len(sys.argv) > 1 else "fridge.jpg").replace("\\", "/")
    agent(f"Here's a photo of my fridge: {photo}. What can I cook tonight?")
