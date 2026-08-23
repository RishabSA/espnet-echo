from pathlib import Path

import pytest
import torch

from scripts.common.audio import load_audio
from scripts.common.whisper_engine import WhisperEngine

pytestmark = pytest.mark.slow

model_id = "openai/whisper-tiny"
audio_file = Path("tests/fixtures/audio/say_sample.wav")


@pytest.fixture(scope="module")
def engine() -> WhisperEngine:
    return WhisperEngine(model_id, device=torch.device("cpu"))


def test_n_best_decode_shape(engine):
    audio = load_audio(str(audio_file))
    hyps = engine.n_best_decode(audio, num_beams=3, num_return_sequences=3)
    assert len(hyps) == 3
    assert [h["rank"] for h in hyps] == [0, 1, 2]
    for hyp in hyps:
        assert hyp["text"]
        assert hyp["tokens"]
        assert abs(hyp["avg_logprob"] - hyp["sum_logprob"] / len(hyp["tokens"])) < 1e-9
        for token in hyp["tokens"]:
            assert not token["tok"].startswith("<|")


def test_n_best_beams_are_distinct(engine):
    # regression: transformers 5's Whisper generate wrapper returns num_beams copies
    # of the top beam; the vanilla GenerationMixin path must give distinct beams
    audio = load_audio(str(audio_file))
    hyps = engine.n_best_decode(audio, num_beams=4, num_return_sequences=4)
    token_seqs = {tuple(t["id"] for t in h["tokens"]) for h in hyps}
    assert len(token_seqs) > 1
    scores = [h["beam_score"] for h in hyps]
    assert scores == sorted(scores, reverse=True)


def test_biased_decode_prompt_never_leaks(engine):
    audio = load_audio(str(audio_file))
    hyps = engine.biased_decode(
        audio, "Kowalski, Zelmark", num_beams=2, num_return_sequences=2
    )
    assert len(hyps) == 2
    for hyp in hyps:
        assert hyp["text"]
        # the prompt prefix lives before <|startoftranscript|> and must never be
        # echoed into the transcript text
        assert not hyp["text"].startswith("Kowalski, Zelmark")


def test_score_text_prefers_own_output(engine):
    # the M0.6 round-trip acceptance: the model's own greedy output must outscore a
    # corrupted variant of it under teacher forcing
    audio = load_audio(str(audio_file))
    hyps = engine.n_best_decode(audio, num_beams=1, num_return_sequences=1)
    own = hyps[0]["text"]

    words = own.split()
    victim = max(range(len(words)), key=lambda i: len(words[i]))
    words[victim] = "flurbish"
    corrupted = " ".join(words)

    own_score = engine.score_text(audio, own)
    bad_score = engine.score_text(audio, corrupted)
    assert own_score["mean_all"] > bad_score["mean_all"]


def test_score_text_focus(engine):
    audio = load_audio(str(audio_file))
    hyps = engine.n_best_decode(audio, num_beams=1, num_return_sequences=1)
    own = hyps[0]["text"]

    words = own.split()
    victim = max(range(len(words)), key=lambda i: len(words[i]))
    start = len(" ".join(words[:victim])) + (1 if victim else 0)
    end = start + len(words[victim])

    scored = engine.score_text(audio, own, focus=(start, end))
    assert scored["n_focus_tokens"] >= 1
    assert scored["n_focus_tokens"] < scored["n_tokens"]
