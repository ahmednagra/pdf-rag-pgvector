from app.answer import NO_ANSWER
from app.llm import LLMResponse
from tests.fakes import FakeLLM, tool_call
from tests.pdfs import make_pdf

KETTLE = make_pdf(
    [
        "The kettle battery charges in four hours of full sun.",
        "Error code E3 means the water level is below the MIN line.",
    ]
)


def upload(client, name="kettle.pdf", data=KETTLE):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_upload_list_and_delete(make_client):
    client = make_client()
    first = upload(client)
    assert first.status_code == 200 and first.json()["status"] == "created"
    assert upload(client).json()["status"] == "unchanged"

    docs = client.get("/documents").json()
    assert [d["filename"] for d in docs] == ["kettle.pdf"]

    assert client.delete(f"/documents/{docs[0]['id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{docs[0]['id']}").status_code == 404


def test_upload_validation(make_client, settings):
    client = make_client()
    assert upload(client, name="notes.txt").status_code == 422
    assert upload(client, data=b"not a pdf").status_code == 422
    settings.max_upload_bytes = 10
    assert upload(client).status_code == 413


def test_search_endpoint(make_client):
    client = make_client()
    upload(client)
    hits = client.post("/search", json={"query": "error code E3", "top_k": 1}).json()
    assert hits[0]["page_number"] == 2
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 1000}).status_code == 422


def test_answer_with_llm_returns_resolved_citations(make_client):
    llm = FakeLLM([LLMResponse("E3 means the water is below the MIN line [1]. Also [9].")])
    client = make_client(llm)
    upload(client)
    body = client.post("/answer", json={"question": "What does error E3 mean?"}).json()

    assert [c["source"] for c in body["citations"]] == [1]  # [9] is not a real source: dropped
    assert body["citations"][0]["page_number"] == 2
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "[1] (kettle.pdf, page 2)" in prompt
    assert "Question: What does error E3 mean?" in prompt


def test_answer_without_documents_skips_the_llm(make_client):
    llm = FakeLLM([])
    body = make_client(llm).post("/answer", json={"question": "anything?"}).json()
    assert body["answer"] == NO_ANSWER and body["citations"] == []
    assert llm.calls == []


def test_extractive_baseline_cites_sources(make_client):
    client = make_client(None)
    upload(client)
    body = client.post("/answer", json={"question": "What does error code E3 mean?"}).json()
    assert "E3" in body["answer"]
    assert body["citations"] and body["citations"][0]["page_number"] == 2


def test_chat_runs_tool_loop(make_client):
    llm = FakeLLM(
        [
            tool_call("list_documents", call_id="c1"),
            tool_call("search_documents", call_id="c2", query="error code E3", top_k=2),
            LLMResponse("It means the water is below the MIN line [1]."),
        ]
    )
    client = make_client(llm)
    upload(client)
    body = client.post("/chat", json={"question": "What is E3?"}).json()

    assert [t["tool"] for t in body["tool_trace"]] == ["list_documents", "search_documents"]
    assert body["citations"][0]["page_number"] == 2
    # The tool result was sent back to the model, linked to the right call id.
    tool_messages = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]
    assert "kettle.pdf" in tool_messages[0]["content"]


def test_chat_tool_loop_is_bounded(make_client, settings):
    settings.max_tool_rounds = 2
    looping = [tool_call("search_documents", call_id=f"c{i}", query="kettle") for i in range(2)]
    llm = FakeLLM([*looping, LLMResponse("Final answer [1].")])
    client = make_client(llm)
    upload(client)
    body = client.post("/chat", json={"question": "loop forever"}).json()

    assert len(llm.calls) == 3
    assert llm.calls[-1]["tool_choice"] == "none"  # last round forces a text answer
    assert "Do not call tools" in llm.calls[-1]["messages"][-1]["content"]
    assert body["answer"] == "Final answer [1]."


def test_chat_handles_unknown_tool_and_bad_arguments(make_client):
    llm = FakeLLM(
        [
            tool_call("delete_everything", call_id="c1"),
            tool_call("search_documents", call_id="c2", query=""),
            LLMResponse(NO_ANSWER),
        ]
    )
    client = make_client(llm)
    upload(client)
    body = client.post("/chat", json={"question": "?"}).json()
    assert [t["result_summary"] for t in body["tool_trace"]] == [
        "unknown tool",
        "invalid arguments",
    ]
    assert body["citations"] == []


def test_chat_requires_real_llm(make_client):
    assert make_client(None).post("/chat", json={"question": "hi"}).status_code == 400
