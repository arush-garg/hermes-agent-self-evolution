from unittest.mock import patch

from evolution.core.pairwise import CallBudget, LLMClient


class FakeLM:
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, messages):
        if self.temperature is not None:
            raise RuntimeError("BadGatewayError: claude/claude-sonnet-5-5: model — `temperature` is deprecated for this model. (HTTP 400)")
        return ["ok-without-temperature"]


def test_retries_without_temperature_when_model_rejects_it():
    made = []

    def fake_make(self, model, temperature="__default__"):
        t = self.temperature if temperature == "__default__" else temperature
        made.append(t)
        return FakeLM(t)

    with patch.object(LLMClient, "_make", fake_make):
        c = LLMClient(CallBudget(10), "optimizer", temperature=0.7, model="openai/optimizer")
        assert c.chat("hi") == "ok-without-temperature"
        assert made == [0.7, None]
        # sticky: the next call reuses the temperature-less LM
        assert c.chat("again") == "ok-without-temperature"
        assert made == [0.7, None]


def test_other_errors_still_raise():
    class Boom:
        def __call__(self, messages):
            raise RuntimeError("HTTP 504 timed out")

    with patch.object(LLMClient, "_make", lambda self, m, temperature="__default__": Boom()):
        c = LLMClient(CallBudget(10), "optimizer", temperature=0.7, model="openai/optimizer")
        try:
            c.chat("hi")
            assert False
        except RuntimeError as e:
            assert "504" in str(e)
