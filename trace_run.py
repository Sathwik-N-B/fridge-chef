"""Run the agent once and print every step, so you can see how the pieces connect.

    python trace_run.py photos/fridge_5.jpg
"""
import json
import os
import sys
import time

os.environ["FRIDGE_TRACE"] = "1"  # turns on the log lines inside tools.py

from agent_after import agent  # noqa: E402  (import after the env var is set)

photo = (sys.argv[1] if len(sys.argv) > 1 else "fridge.jpg").replace("\\", "/")
prompt = f"Here's a photo of my fridge: {photo}. What can I cook tonight?"
t0 = time.time()


def stamp() -> str:
    return f"[{time.time() - t0:5.1f}s]"


seen_tool_calls = set()


def handler(**event):
    """Strands calls this for every streaming event from the agent loop."""
    tool_use = event.get("current_tool_use")
    if tool_use and tool_use.get("toolUseId") not in seen_tool_calls and tool_use.get("name"):
        seen_tool_calls.add(tool_use["toolUseId"])
        print(f"\n{stamp()} AGENT -> TOOL: model decided to call '{tool_use['name']}'",
              file=sys.stderr, flush=True)
    if event.get("start_event_loop"):
        print(f"\n{stamp()} LOOP: a new model call starts", file=sys.stderr, flush=True)


agent.callback_handler = handler

print(f"{stamp()} USER PROMPT: {prompt}", file=sys.stderr)
print(f"{stamp()} SYSTEM PROMPT: {agent.system_prompt[:120]}...", file=sys.stderr)
print(f"{stamp()} TOOLS GIVEN TO MODEL: {list(agent.tool_names)}", file=sys.stderr)
result = agent(prompt)

print("\n" + "=" * 70, file=sys.stderr)
print("FULL CONVERSATION (what the model actually saw and produced)", file=sys.stderr)
print("=" * 70, file=sys.stderr)
for i, msg in enumerate(agent.messages, 1):
    for block in msg["content"]:
        if "text" in block:
            print(f"{i}. {msg['role'].upper()} text: {block['text']}", file=sys.stderr)
        elif "toolUse" in block:
            tu = block["toolUse"]
            print(f"{i}. {msg['role'].upper()} tool call: {tu['name']}({json.dumps(tu['input'])})",
                  file=sys.stderr)
        elif "toolResult" in block:
            tr = block["toolResult"]["content"]
            print(f"{i}. {msg['role'].upper()} tool result: {tr}", file=sys.stderr)

m = result.metrics
print(f"\nSUMMARY: {m.cycle_count} model calls, {sum(m.tool_metrics[t].call_count for t in m.tool_metrics)} "
      f"tool call(s), tokens={m.accumulated_usage}, total {time.time() - t0:.1f}s", file=sys.stderr)
