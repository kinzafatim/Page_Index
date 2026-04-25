import pymupdf
doc = pymupdf.open('UCIE_1.1.pdf')
toc = doc.get_toc()
print(f"Number of TOC entries: {len(toc)}")
if len(toc) > 0:
    print(toc[:10])
