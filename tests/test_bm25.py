from catalogue_rag.bm25 import BM25, looks_like_code, tokenize


def test_codes_stay_whole_and_also_split():
    toks = tokenize("Order CH/10/900 or C700HOSIL")
    assert "ch/10/900" in toks and "900" in toks
    assert "c700hosil" in toks and "700" in toks and "hosil" in toks


def test_stop_words_are_dropped():
    assert tokenize("What is the battery of the handle") == ["battery", "handle"]


def test_looks_like_code():
    assert looks_like_code("c700hosil") and looks_like_code("ch/10/900") and looks_like_code("h100")
    assert not looks_like_code("battery") and not looks_like_code("55")


def test_rare_terms_outrank_common_ones():
    docs = ["door handle satin chrome", "door handle black", "door closer with hold open arm", "door handle IP42 rated"]
    best = BM25(docs).search("IP42 door", 1)[0][0]
    assert best == 3


def test_model_number_finds_its_variant_code():
    docs = ["Hold-Open | C700HOSIL | Silver", "Standard arm | C726SIL | Silver", "Padbolt 100mm"]
    assert BM25(docs).search("C700 closer in silver", 1)[0][0] == 0


def test_unknown_terms_return_nothing():
    assert BM25(["door handle"]).search("zzzz", 5) == []
