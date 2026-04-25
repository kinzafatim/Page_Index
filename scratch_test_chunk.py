import math
import json
import os
import time
from pageindex.utils import ConfigLoader, llm_completion, extract_json

def toc_transformer_chunked(toc_content, model=None):
    lines = toc_content.split('\n')
    chunk_size = 50
    
    init_prompt = """
    You are given a partial table of contents. Your job is to transform this partial table of content into a JSON array included in `table_of_contents`.

    The `structure` is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index "1", the first subsection has structure index "1.1", the second subsection has structure index "1.2", etc. If no structure index is found, use null.

    EXAMPLE INPUT:
    3.0 Die-to-Die Adapter : 43
    3.1 Stack Multiplexing : 44
    3.2 Link Initialization: 46

    EXAMPLE OUTPUT:
    ```json
    {
        "thinking": "I see three items. 1. '3.0 Die-to-Die Adapter' on page 43. 2. '3.1 Stack Multiplexing' on page 44. 3. '3.2 Link Initialization' on page 46.",
        "table_of_contents": [
            {
                "structure": "3.0",
                "title": "Die-to-Die Adapter",
                "page": 43
            },
            {
                "structure": "3.1",
                "title": "Stack Multiplexing",
                "page": 44
            },
            {
                "structure": "3.2",
                "title": "Link Initialization",
                "page": 46
            }
        ]
    }
    ```

    Now, process the following partial table of contents. Do not output anything except the JSON format.
    """

    chunk = '\n'.join(lines[0:chunk_size])
    prompt = init_prompt + '\n Given partial table of contents\n:\n' + chunk
    
    print("Sending prompt of length:", len(prompt))
    response = llm_completion(model=model, prompt=prompt)
    print("Response preview:")
    print(response[:500])

if __name__ == "__main__":
    with open('logs/UCIE_1.1.pdf_20260424_203418.json', 'r') as f:
        data = json.load(f)
        toc_content = data[2]['toc_content']
        toc_transformer_chunked(toc_content, model="openai/gemma-4-31b")
