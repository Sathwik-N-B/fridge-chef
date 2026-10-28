# Fridge Chef

A recipe agent that can look at a photo of your fridge. Built with [Strands Agents](https://strandsagents.com) and Amazon Bedrock (Nova Lite).

## What it shows
The same prompt, with and without a tool:

> "Here's a photo of my fridge: photos/fridge.jpg. What can I cook tonight?"

- **Before** (`agent_before.py`): no tools. The model says it can't see images and invents generic recipes.
- **After** (`agent_after.py`): the agent calls `identify_ingredients`, reads the detected items, and suggests recipes from them.

## How it works
```
user prompt -> agent (model call 1) -> decides to call identify_ingredients(path)
            -> tool: open image, shrink to JPEG, ask a Bedrock vision model for the food items
            -> agent (model call 2) reads the tool result -> answers with recipes
```
- `model.py`: picks the LLM (Bedrock Nova Lite by default).
- `tools.py`: `identify_ingredients` (Bedrock vision) and `detect_scene_labels` (Amazon Rekognition, kept for comparison).
- `trace_run.py`: runs the agent once and prints every step, including the tool's internal steps.

## Lessons from building it
- Amazon Rekognition `DetectLabels` describes the *scene* ("Refrigerator", "Appliance"), not each item on the shelves. A vision LLM listed the contents far better, so the main tool uses Nova Lite vision.
- Windows paths like `photos\fridge.jpg` can lose characters when an LLM passes them as JSON (`\f` becomes a form feed). The scripts use forward slashes and the tool repairs damaged paths.
- Decoding at temperature 0 can make the vision model loop on repeated items until it hits `maxTokens`; the trace logs `stopReason` to show it.

## Run it
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
aws configure            # needs Bedrock model access and (for the scene tool) rekognition:DetectLabels

python agent_before.py photos/fridge.jpg
python agent_after.py  photos/fridge.jpg
python trace_run.py    photos/fridge.jpg    # step-by-step trace
```
Set `AWS_REGION` if your Bedrock/Rekognition access is not in `us-east-2`. Never commit access keys.

## Limitations
Vision results can miss items or add wrong ones, so the agent is told to ask the user to confirm the list.
