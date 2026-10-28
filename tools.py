"""Tools for Fridge Chef.

identify_ingredients: reads the items in a photo with a vision model on Amazon Bedrock.
detect_scene_labels: the Amazon Rekognition version, kept for comparison. It labels the
scene (e.g. "Refrigerator") rather than every item on the shelves.
"""
import io
import os
import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image
from strands import tool

REGION = os.environ.get("AWS_REGION", "us-east-2")
VISION_MODEL_ID = "us.amazon.nova-lite-v1:0"
MAX_SIDE = 1568  # pixels; large photos are shrunk before upload

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
rekognition = boto3.client("rekognition", region_name=REGION)

def _trace(msg: str) -> None:
    """Print a step-by-step log line when FRIDGE_TRACE=1 (see trace_run.py)."""
    if os.environ.get("FRIDGE_TRACE") == "1":
        print(f"      [tool] {msg}", file=sys.stderr, flush=True)


# Generic labels that are not useful as ingredients (used by the Rekognition tool).
IGNORE = {
    "food", "produce", "plant", "fruit", "vegetable", "refrigerator", "fridge",
    "appliance", "shelf", "furniture", "indoors", "drink", "beverage", "bottle",
    "container", "box", "bag", "package", "meal", "dish", "pantry", "cupboard",
    "device", "electrical device", "interior design", "kitchen",
}


# A Windows path like photos\fridge.jpg can reach a tool with "\f" turned into a
# form-feed character (JSON escape). Turn those control characters back into "\<letter>".
_ESCAPES = {"\x07": "\\a", "\x08": "\\b", "\x0c": "\\f", "\n": "\\n", "\r": "\\r",
            "\t": "\\t", "\x0b": "\\v"}


def _repair_path(path: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in path)


def _load_jpeg(image_path: str) -> bytes:
    """Open any common image (JPG/PNG/WebP), shrink it, and return JPEG bytes."""
    path = _repair_path(image_path)
    _trace(f"1. opening file: {path!r}")
    with Image.open(path) as img:
        _trace(f"2. original image: {img.format}, {img.size[0]}x{img.size[1]}, {os.path.getsize(path) // 1024} KB")
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        _trace(f"3. converted to JPEG: {img.size[0]}x{img.size[1]}, {len(buf.getvalue()) // 1024} KB")
        return buf.getvalue()


@tool
def identify_ingredients(image_path: str) -> str:
    """Look at a photo of a fridge, pantry, or counter and list the food items visible.

    Use this whenever the user mentions a photo or image file. Pass the local
    file path. Returns a comma-separated list of the food and drink items seen.

    Args:
        image_path: Local path to an image (JPG, PNG, or WebP).
    """
    try:
        image_bytes = _load_jpeg(image_path)
    except FileNotFoundError:
        return f"Error: could not find an image at '{image_path}'."
    except Exception as e:  # not an image, corrupt file
        return f"Error: could not read '{image_path}' as an image ({e})."

    _trace(f"4. sending image to Bedrock ({VISION_MODEL_ID}, region {REGION})")
    started = time.time()
    try:
        resp = bedrock.converse(
            modelId=VISION_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
                    {"text": (
                        "List the specific food and drink items visible in this photo, "
                        "including fresh produce, packaged goods, jars, cans, and boxes. "
                        "Name specific items (e.g. 'apples', 'tomato sauce'), not categories "
                        "like 'vegetables' or 'dairy'. Comma-separated, each item once, no "
                        "extra text. Only list items you can clearly see; do not guess or add "
                        "items that are typically in a fridge, and ignore text on labels. "
                        "Only if there is truly no food or drink in the photo, "
                        "reply exactly: NONE"
                    )},
                ],
            }],
            inferenceConfig={"maxTokens": 300, "temperature": 0},
        )
    except (ClientError, BotoCoreError) as e:
        return f"Error calling Bedrock: {e}"

    raw = resp["output"]["message"]["content"][0]["text"].strip()
    usage = resp.get("usage", {})
    stop = resp.get("stopReason")
    _trace(f"5. vision model replied in {time.time() - started:.1f}s "
           f"(tokens in={usage.get('inputTokens')}, out={usage.get('outputTokens')}, "
           f"stopReason={stop}{' <-- CUT OFF at maxTokens' if stop == 'max_tokens' else ''})")
    _trace(f"6. raw reply: {raw!r}")
    if raw.upper().startswith("NONE"):
        return "No food items detected. Ask the user to list them."

    seen, items = set(), []
    for part in raw.replace("\n", ",").split(","):
        item = part.strip().strip(".").lower()
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    _trace(f"7. cleaned list ({len(items)} unique items) returned to the agent")
    return "Detected items: " + ", ".join(items)


@tool
def detect_scene_labels(image_path: str, min_confidence: float = 80.0) -> str:
    """Label the overall scene of a photo with Amazon Rekognition (e.g. 'Refrigerator').

    This describes what kind of scene it is, not each item on the shelves. Use
    identify_ingredients to list food items.

    Args:
        image_path: Local path to an image (JPG, PNG, or WebP).
        min_confidence: Minimum Rekognition confidence (0-100) to keep a label.
    """
    try:
        image_bytes = _load_jpeg(image_path)
    except FileNotFoundError:
        return f"Error: could not find an image at '{image_path}'."
    except Exception as e:
        return f"Error: could not read '{image_path}' as an image ({e})."

    try:
        resp = rekognition.detect_labels(
            Image={"Bytes": image_bytes}, MaxLabels=40, MinConfidence=min_confidence
        )
    except (ClientError, BotoCoreError) as e:
        return f"Error calling Rekognition: {e}"

    items = [
        f"{label['Name']} ({label['Confidence']:.0f}%)"
        for label in resp["Labels"]
        if label["Name"].lower() not in IGNORE
    ]
    return "Scene labels: " + ", ".join(items) if items else "No useful labels found."
