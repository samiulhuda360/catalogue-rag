from catalogue_rag.bm25 import BM25
from catalogue_rag.chunking import Chunk
from catalogue_rag.generation import NOT_FOUND, Generator, format_sources, parse_citations
from catalogue_rag.index import Index
from catalogue_rag.retrieval import Hit, Retriever, rrf


def make_index(chunks: list[Chunk]) -> Index:
    return Index(chunks, {c.id: i for i, c in enumerate(chunks)}, BM25([c.search_text() for c in chunks]), None)


CHUNKS = [
    Chunk("a::p1::c0", "a", "Acme H200", 1, "Scope", "1 pair of handles with battery (Lithium CR123A)"),
    Chunk("a::p1::c1", "a", "Acme H200", 1, "Technical data", "Battery life | 30 openings per day for 50 months"),
    Chunk("b::p4::c0", "b", "Closers", 4, "C700", "Hold-Open | C700HOSIL | Silver"),
    Chunk("b::p9::c1", "b", "Closers", 9, "7726", "Standard arm | C726SIL | Silver"),
]


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf({"bm25": ["x", "y", "z"], "vector": ["y", "x", "w"]}, {"bm25": 1, "vector": 1})
    assert [cid for cid, _, _ in fused][:2] in (["x", "y"], ["y", "x"])
    assert fused[-1][0] in ("z", "w")


def test_rrf_weight_shifts_the_order():
    fused = rrf({"bm25": ["x", "y"], "vector": ["y", "x"]}, {"bm25": 2.0, "vector": 1.0})
    assert fused[0][0] == "x"


def test_short_pages_are_returned_whole():
    hits = Retriever(make_index(CHUNKS)).search("Acme H200 battery", top_k=4, mode="bm25")
    page = next(h for h in hits if h.chunk.doc == "a")
    assert "CR123A" in page.chunk.text and "50 months" in page.chunk.text
    assert sum(1 for h in hits if h.chunk.doc == "a") == 1  # the page appears once


def test_part_number_query_finds_the_row():
    hits = Retriever(make_index(CHUNKS)).search("C700 hold open closer silver", top_k=1, mode="bm25")
    assert "C700HOSIL" in hits[0].chunk.text


def test_citations_are_parsed_and_bounded():
    assert parse_citations("Uses CR123A [1]. Lasts 50 months [2][1]. See [3, 9].", n_sources=3) == [1, 2, 3]


def test_sources_are_numbered_with_title_page_and_section():
    text = format_sources([Hit(CHUNKS[1], 1.0)])
    assert text.startswith("[1] Acme H200 - page 1 - Technical data")


class FakeLLM:
    """Stands in for the OpenAI client: returns a fixed reply, records the prompt."""

    def __init__(self, reply: str):
        self.reply, self.prompts = reply, []
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.prompts.append(kw["messages"])

        class Msg:
            content = self.reply

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]
            usage = None

        return Resp()


def generator_with(reply: str) -> Generator:
    g = Generator.__new__(Generator)
    g.client, g.model, g.temperature, g.max_tokens = FakeLLM(reply), "fake", 0.0, 100
    return g


def test_not_found_becomes_a_decline_without_citations():
    out = generator_with(f"{NOT_FOUND}\nThe extracts cover batteries only.").answer("Price?", [Hit(CHUNKS[0], 1.0)])
    assert out.declined and out.cited == [] and NOT_FOUND not in out.text


def test_answers_report_what_they_cite():
    out = generator_with("It uses a CR123A battery [1].").answer("Battery?", [Hit(CHUNKS[0], 1.0), Hit(CHUNKS[1], 0.5)])
    assert not out.declined and out.cited == [1]


def test_no_sources_means_no_model_call():
    g = generator_with("should not be used")
    out = g.answer("anything", [])
    assert out.declined and g.client.prompts == []


class FakeStreamLLM(FakeLLM):
    """Returns the reply in small pieces, like a streaming API."""

    def create(self, **kw):
        if not kw.get("stream"):
            return super().create(**kw)
        self.prompts.append(kw["messages"])

        def pieces():
            for i in range(0, len(self.reply), 4):
                class Delta:
                    content = self.reply[i:i + 4]

                class Choice:
                    delta = Delta()

                class Event:
                    choices = [Choice()]

                yield Event()

        return pieces()


def stream_with(reply: str) -> list:
    g = Generator.__new__(Generator)
    g.client, g.model, g.temperature, g.max_tokens = FakeStreamLLM(reply), "fake", 0.0, 100
    return list(g.stream("q", [Hit(CHUNKS[0], 1.0), Hit(CHUNKS[1], 0.5)]))


def test_stream_sends_deltas_then_a_cleaned_answer():
    events = stream_with("It uses a CR123A battery [1].")
    text = "".join(v for k, v in events if k == "delta")
    assert text == "It uses a CR123A battery [1]."
    kind, final = events[-1]
    assert kind == "done" and final.cited == [1] and not final.declined


def test_stream_never_shows_the_decline_marker():
    events = stream_with(f"{NOT_FOUND} The extracts cover batteries, not prices.")
    kinds = [k for k, _ in events]
    shown = "".join(v for k, v in events if k == "delta")
    assert kinds[0] == "declined" and NOT_FOUND not in shown
    assert events[-1][1].declined and events[-1][1].text == "The extracts cover batteries, not prices."


def test_uncited_refusal_without_marker_is_a_decline():
    from catalogue_rag.generation import finish

    g = finish("The extracts do not mention AS 1428.1 certification for the L30 lever set.", 8, "m")
    assert g.declined and g.cited == []
    g = finish("The L30 is fire rated to 60 minutes [2]. The extracts do not mention its warranty.", 8, "m")
    assert not g.declined and g.cited == [2]  # cited answer with a noted gap stays an answer
    g = finish("It uses a CR123A battery [1].", 8, "m")
    assert not g.declined
