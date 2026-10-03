from catalogue_rag.chunking import chunk_document, table_to_rows

DOC = """# sample.pdf

**Brand:** Acme
**Parsed:** 2026-10-03

---

# Acme H200 Handle

## Product description

* Wireless handle for interior doors
* Works with standard mortice locks

## Technical data

<table><tr><td>Battery</td><td>1 x Lithium CR123A</td></tr>
<tr><td>Battery life</td><td>30 openings per day<br/>for 50 months</td></tr>
<tr><td>Class of protection</td><td>IP42</td></tr></table>

---

# Ordering

<table><tr><th>Finish</th><th>Part number</th></tr>
<tr><td>Satin chrome</td><td>H200SC</td></tr>
<tr><td>Black</td><td>H200BLK</td></tr></table>
"""


def test_header_is_not_content_and_pages_are_numbered():
    chunks = chunk_document("sample", DOC)
    assert all("Parsed" not in c.text for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


def test_document_title_comes_from_first_real_heading():
    assert chunk_document("sample", DOC)[0].title == "Acme H200 Handle"


def test_tables_become_rows_and_line_breaks_survive():
    rows = table_to_rows("<table><tr><td>Battery life</td><td>30 a day<br/>for 50 months</td></tr></table>")
    assert rows == ["Battery life | 30 a day; for 50 months"]


def test_table_rows_keep_their_section_breadcrumb():
    battery = next(c for c in chunk_document("sample", DOC) if "CR123A" in c.text)
    assert battery.section == "Technical data"
    assert battery.search_text().startswith("Acme H200 Handle > Technical data")


def test_section_carries_over_to_the_next_page():
    ordering = next(c for c in chunk_document("sample", DOC) if "H200BLK" in c.text)
    assert ordering.page == 2 and ordering.section == "Ordering"


def test_big_tables_split_by_row_with_the_header_repeated():
    rows = "".join(f"<tr><td>Part {i}</td><td>CODE-{i:04d}</td></tr>" for i in range(80))
    doc = f"# t.pdf\n\n---\n\n# Big\n\n<table><tr><th>Name</th><th>Code</th></tr>{rows}</table>"
    chunks = chunk_document("big", doc, max_chars=300)
    assert len(chunks) > 3
    assert all(c.text.splitlines()[0] == "Name | Code" for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    codes = [line for c in chunks for line in c.text.splitlines()[1:]]
    assert len(codes) == 80  # nothing lost, nothing duplicated


def test_chunk_ids_are_unique():
    ids = [c.id for c in chunk_document("sample", DOC)]
    assert len(ids) == len(set(ids))
