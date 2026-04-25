"""
Agentic Vectorless RAG for local llama.cpp (Gemma-4-31b)

Architecture:
- Uses start_index/end_index from tree nodes to fetch real PDF page text via PyMuPDF
- Full drill-down: get_structure(null) → top chapters, get_structure(node_id) → sub-sections
- Multi-node fetching: get_node_text(["id1", "id2"]) pulls pages for all selected nodes
- Handles both standard JSON and Gemma's native <|tool_call> tokens
- Loop runs up to max_steps; model calls final_answer when done
"""

import json
import re
import litellm
import fitz  # PyMuPDF

# ─── Config ──────────────────────────────────────────────────────────────────
litellm.api_base = "http://localhost:8080/v1"
litellm.api_key = "sk-no-key-required"
MODEL = "openai/gemma-4-31b"
PDF_PATH = "./UCIE_1.1.pdf"
STRUCTURE_PATH = "./results/UCIE_1.1_structure.json"
MAX_STEPS = 10

# Thinking control:
# - False = fast JSON mode (navigation steps: get_structure, get_node_text)
# - True  = deep reasoning (only for the final_answer synthesis call)
THINKING_FOR_NAVIGATION = False
THINKING_FOR_FINAL_ANSWER = True

# ─── PDF page reader ─────────────────────────────────────────────────────────

def read_pdf_pages(pdf_path: str, start_page: int, end_page: int) -> str:
    """Read pages start_page to end_page (1-indexed, inclusive) from the PDF."""
    doc = fitz.open(pdf_path)
    total = len(doc)
    texts = []
    for p in range(start_page, end_page + 1):
        if 1 <= p <= total:
            texts.append(doc[p - 1].get_text())
    doc.close()
    return "\n".join(texts)

# ─── Tree helpers ─────────────────────────────────────────────────────────────

def load_tree(json_path: str) -> list:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('structure', data) if isinstance(data, dict) else data


def find_node(nodes: list, target_id: str) -> dict | None:
    for node in nodes:
        if node.get('node_id') == target_id:
            return node
        if node.get('nodes'):
            res = find_node(node['nodes'], target_id)
            if res:
                return res
    return None


def format_structure(nodes: list) -> str:
    """Format a flat list of nodes for the LLM (no recursion — one level only)."""
    lines = []
    for node in nodes:
        child_count = len(node.get('nodes', []))
        child_hint = f"  [{child_count} sub-sections]" if child_count else ""
        lines.append(
            f"- [Node ID: {node['node_id']}] {node['title']} "
            f"(pages {node['start_index']}–{node['end_index']}){child_hint}\n"
            f"  Summary: {node.get('summary', '')[:300]}"
        )
    return "\n".join(lines)


def get_node_pages(nodes: list, target_id: str, pdf_path: str) -> str:
    """Fetch actual PDF page text for a node using its start/end_index."""
    node = find_node(nodes, target_id)
    if not node:
        return f"Error: Node ID '{target_id}' not found in the tree."
    start = node.get('start_index', 1)
    end = node.get('end_index', start)
    # Clamp range to avoid huge context — max 5 pages per node
    end = min(end, start + 4)
    text = read_pdf_pages(pdf_path, start, end)
    return f"[{node['title']} | Pages {start}–{end}]\n{text}"

# ─── LLM output parser ────────────────────────────────────────────────────────

def parse_action(content: str) -> dict | None:
    """
    Parse the LLM response. Handles:
      1. Standard ```json {...} ``` blocks
      2. Gemma native <|tool_call>call:tool_name{...}<tool_call|>
    """
    # 1. Standard JSON block
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match and '<|tool_call>' not in content:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # 2. Native <|tool_call> format: <|tool_call>call:tool_name{args}<tool_call|>
    native = re.search(r'<\|tool_call>call:([^\s{<]+)\s*(\{.*?)(?:<tool_call\|>|$)', content, re.DOTALL)
    if native:
        tool_name = native.group(1).strip()
        args_str = native.group(2).strip()
        # Fix unquoted keys: {node_ids:[...]} → {"node_ids":[...]}
        args_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', args_str)
        try:
            args = json.loads(args_str)
        except Exception:
            args = {}
        return {"reasoning": "(native tool_call token)", "tool": tool_name, "args": args}

    return None

# ─── Main agentic loop ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an elite hardware architecture agent with access to 3 tools.
Use them step-by-step to answer the user's question accurately.

TOOLS:
1. get_structure
   - args: {"node_id": null}           → returns the root-level chapters
   - args: {"node_id": "0012"}         → returns sub-sections of that chapter
   Use this first to navigate the document hierarchy before reading any text.

2. get_node_text
   - args: {"node_ids": ["0012", "0015"]}  → returns the actual PDF text for those nodes
   You can request multiple nodes at once. Prefer specific sub-section nodes over root chapters.

3. final_answer
   - args: {"answer": "Your answer here"}
   Call this ONLY when you have enough text to answer fully.

RESPONSE FORMAT (always one of these, nothing else):
```json
{"reasoning": "<why you are calling this tool>", "tool": "get_structure", "args": {"node_id": null}}
```
```json
{"reasoning": "<why you chose these nodes>", "tool": "get_node_text", "args": {"node_ids": ["0012"]}}
```
```json
{"reasoning": "<why you are done>", "tool": "final_answer", "args": {"answer": "<full answer>"}}
```

RULES:
- Always call get_structure first to discover real Node IDs. Never guess Node IDs.
- Always output only a single JSON block. No extra text, no <|tool_call> tokens.
- If text returned is empty or unhelpful, drill down further with get_structure before retrying.
"""


def query_agentic_rag(question: str):
    nodes = load_tree(STRUCTURE_PATH)
    print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    for step in range(MAX_STEPS):
        print(f"\n[Step {step + 1}/{MAX_STEPS}] Waiting for model...")
        thinking_on = THINKING_FOR_NAVIGATION  # default: fast mode
        response = litellm.completion(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking_on}}
        )

        content = response.choices[0].message.content.strip()
        print(f"\n[DEBUG] Model output:\n{content}\n")

        action = parse_action(content)
        if not action:
            # Model skipped the JSON format and answered directly — treat as final answer
            print("[INFO] Model answered directly without a tool call. Treating as final answer.")
            print(f"\n{'='*60}\nFINAL ANSWER\n{'='*60}\n{content}\n{'='*60}")
            return content

        reasoning = action.get("reasoning", "")
        tool = action.get("tool", "")
        args = action.get("args", {})

        print(f"  Reasoning : {reasoning[:200]}")
        print(f"  Tool      : {tool}")
        print(f"  Args      : {args}")

        # Append assistant turn
        messages.append({"role": "assistant", "content": content})

        # ── Tool dispatch ──────────────────────────────────────────────────────
        if tool == "get_structure":
            node_id = args.get("node_id")
            if node_id is None:
                target_nodes = nodes
            else:
                target_node = find_node(nodes, node_id)
                if not target_node:
                    tool_output = f"Error: Node '{node_id}' not found."
                else:
                    target_nodes = target_node.get('nodes', [])
                    if not target_nodes:
                        tool_output = f"Node '{node_id}' has no sub-sections. Read it directly with get_node_text."
                        messages.append({"role": "user", "content": f"[Tool: get_structure] {tool_output}"})
                        continue

            tool_output = format_structure(target_nodes)
            print(f"  -> Returned {len(target_nodes)} nodes to model")
            messages.append({"role": "user", "content": f"[Tool: get_structure]\n{tool_output}"})

        elif tool == "get_node_text":
            node_ids = args.get("node_ids", [])
            if not node_ids:
                tool_output = "Error: node_ids list is empty."
            else:
                parts = []
                for nid in node_ids:
                    parts.append(get_node_pages(nodes, nid, PDF_PATH))
                tool_output = "\n\n".join(parts)
                print(f"  -> Returned {len(tool_output)} characters of PDF text")
            messages.append({"role": "user", "content": f"[Tool: get_node_text]\n{tool_output}"})

        elif tool == "final_answer":
            # Re-run the synthesis with thinking ENABLED for deeper, more accurate reasoning
            if THINKING_FOR_FINAL_ANSWER:
                print("  [Thinking ON] Re-running final answer synthesis with deep reasoning...")
                final_response = litellm.completion(
                    model=MODEL,
                    messages=messages + [{"role": "user", "content": "Now synthesize a complete, accurate final answer to the original question based on all the information above."}],
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                )
                answer = final_response.choices[0].message.content.strip()
            else:
                answer = args.get("answer", "")
            
            print(f"\n{'='*60}\nFINAL ANSWER\n{'='*60}\n{answer}\n{'='*60}")
            return answer

        else:
            error_msg = f"Error: Tool '{tool}' does not exist. Available tools: get_structure, get_node_text, final_answer."
            messages.append({"role": "user", "content": f"[System] {error_msg}"})

    return "Agent did not produce a final answer within the step limit."


if __name__ == "__main__":
    question = "What are the specific parameters and requirements for the physical layer?"
    query_agentic_rag(question)
