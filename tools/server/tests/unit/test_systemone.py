import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 4096


def test_systemone():
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": {"message": "My card was charged twice."},
        "model": "jev-latest",
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": None, "sales": ["Pricing", "Upgrades"]}},
            "tone": {"type": "score", "criteria": ["Calm", {"level": "Angry"}]},
        },
    })
    assert res.status_code == 200
    answers = res.body["answers"]
    assert list(answers.keys()) == ["refund", "team", "tone"]

    assert answers["refund"]["type"] == "noul"
    assert 0 <= answers["refund"]["noul"] <= 1

    team = answers["team"]
    assert team["type"] == "choice"
    assert list(team["probabilities"].keys()) == ["billing", "tech", "sales"]
    assert sum(team["probabilities"].values()) == pytest.approx(1.0)
    assert team["choice"] == max(team["probabilities"], key=team["probabilities"].get)
    assert 0 <= team["confidence"] <= 1

    tone = answers["tone"]
    assert tone["type"] == "score"
    assert tone["legend"] == {"0": "Calm", "1": {"level": "Angry"}}
    assert tone["score"] == pytest.approx(tone["probabilities"]["1"])
    assert 0 <= tone["confidence"] <= 1

    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 3

    # repeated request must give the same answers from the prompt cache
    res2 = server.make_request("POST", "/v1/systemone", data={
        "state": {"message": "My card was charged twice."},
        "model": "jev-latest",
        "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}},
    })
    assert res2.status_code == 200
    assert res2.body["answers"]["refund"]["noul"] == pytest.approx(answers["refund"]["noul"], abs=1e-3)


def test_systemone_models():
    global server
    server.start()
    res = server.make_request("GET", "/v1/models")
    assert res.status_code == 200
    model = res.body["models"][0]
    assert isinstance(model["name"], str)
    assert isinstance(model["description"], str)
    assert isinstance(model["release_date"], str)


@pytest.mark.parametrize("body,loc", [
    ({"model": "m", "questions": {"q": {"type": "noul"}}}, ["body", "state"]),
    ({"state": "x", "model": "m", "questions": {}}, ["body", "questions"]),
    ({"state": "x", "model": "m", "questions": {"q": {"type": "maybe"}}}, ["body", "questions", "q", "type"]),
    ({"state": "x", "model": "m", "questions": {"q": {"type": "choice"}}}, ["body", "questions", "q", "criteria"]),
    ({"state": "x", "model": "m", "questions": {"q": {"type": "score", "criteria": ["only one"]}}}, ["body", "questions", "q", "criteria"]),
    ({"state": "x", "model": "m", "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(100)}}}}, ["body", "questions", "q", "criteria"]),
])
def test_systemone_invalid(body, loc):
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data=body)
    assert res.status_code == 422
    assert res.body["detail"][0]["loc"] == loc


def test_systemone_permute(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_ARG_SYSTEMONE_PERMUTE", "1")
    server.start()

    def ask(criteria):
        res = server.make_request("POST", "/v1/systemone", data={
            "state": "The package arrived broken.",
            "model": "jev-latest",
            "questions": {"reason": {"type": "choice", "instructions": "What is the issue?", "criteria": criteria}},
        })
        assert res.status_code == 200
        assert res.body["usage"]["output_tokens"] == 2
        return res.body["answers"]["reason"]["probabilities"]

    # both option orders are averaged, so the order in the request does not change the result
    fwd = ask({"damaged": None, "late": None, "wrong_item": None})
    rev = ask({"wrong_item": None, "late": None, "damaged": None})
    for k in fwd:
        assert fwd[k] == pytest.approx(rev[k], abs=1e-3)
