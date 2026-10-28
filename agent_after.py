"""AFTER: same agent, plus one Rekognition tool."""
from strands import Agent

from model import get_model
from tools import identify_ingredients

model = get_model()

agent = Agent(
    model=model,
    tools=[identify_ingredients],
    system_prompt=(
        "You are Fridge Chef, a friendly recipe assistant. "
        "If the user mentions a photo, call identify_ingredients with its path first. "
        "Vision results can miss items or be wrong, so mention that the list may be "
        "incomplete and ask if anything is missing. If no food is detected, ask the user "
        "to list what they have. Then suggest 2-3 simple recipes using the detected items."
    ),
)

if __name__ == "__main__":
    import sys

    # Same prompt as the 'before' run. Record this for the "after".
    photo = (sys.argv[1] if len(sys.argv) > 1 else "fridge.jpg").replace("\\", "/")
    agent(f"Here's a photo of my fridge: {photo}. What can I cook tonight?")
