"""AFTER: same agent, plus one tool that reads a photo."""
from strands import Agent

from model import get_model, strip_thinking
from tools import identify_ingredients

model = get_model()

agent = Agent(
    model=model,
    tools=[identify_ingredients],
    callback_handler=None,  # we print the cleaned answer ourselves
    system_prompt=(
        "You are Fridge Chef, a friendly recipe assistant. "
        "If the user mentions a photo, call identify_ingredients with its path first. "
        "Never write <thinking> tags or show your reasoning. "
        "Then reply in this order: (1) one line listing what was detected, "
        "(2) 2-3 simple recipes that use ONLY detected items plus basic staples "
        "(salt, pepper, oil, water), (3) one short question asking if anything is missing "
        "or wrong. Vision can miss items or misread labels, so do not build a recipe "
        "around a single sauce or syrup. If nothing food-like is detected, skip the "
        "recipes and ask the user to list what they have."
    ),
)

if __name__ == "__main__":
    import sys

    # Same prompt as the 'before' run. Record this for the "after".
    photo = (sys.argv[1] if len(sys.argv) > 1 else "fridge.jpg").replace("\\", "/")
    result = agent(f"Here's a photo of my fridge: {photo}. What can I cook tonight?")
    print(strip_thinking(str(result)))
