import json
from pageindex.utils import ConfigLoader, get_page_tokens
from pageindex.page_index import find_toc_pages, toc_extractor

pdf_path = 'UCIE_1.1.pdf'
opt = ConfigLoader().load({'model': 'openai/gemma-4-31b'})
page_list = get_page_tokens(pdf_path, model=opt.model)
toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)

toc_json = toc_extractor(page_list, toc_page_list, opt.model)
print(f"toc_content length: {len(toc_json['toc_content'])}")
print(f"toc_content lines: {len(toc_json['toc_content'].splitlines())}")
print(toc_json['toc_content'][:500])
