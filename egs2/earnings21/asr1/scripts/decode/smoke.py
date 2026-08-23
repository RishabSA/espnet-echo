import argparse

from scripts.common.audio import load_audio
from scripts.common.phonetics import phones
from scripts.common.whisper_engine import WhisperEngine, pick_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M0.8 smoke test: n-best with logprobs on ~30 s of audio, plus the score_text round trip.")
    parser.add_argument("--audio", type=str, required=True, help="Path to a wav file of roughly 30 s (required).")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3", help="HF model id (default: openai/whisper-large-v3).")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam width (default: 4).")
    parser.add_argument("--num-return", type=int, default=4, help="How many hypotheses to return (default: 4).")
    args = parser.parse_args()

    # force the nltk downloads (g2p_en fetches cmudict + tagger on first use) here
    # rather than mid-experiment, per spec section 3.3
    print(f"g2p smoke: Kowalski -> {' '.join(phones('Kowalski'))}")

    device = pick_device()
    print(f"device: {device}")
    engine = WhisperEngine(args.model, device=device)
    audio = load_audio(args.audio)
    print(f"audio: {audio.shape[0] / 16000:.1f} s")

    hyps = engine.n_best_decode(audio, num_beams=args.num_beams, num_return_sequences=args.num_return)
    for hyp in hyps:
        print(f"[{hyp['rank']}] avg_logprob={hyp['avg_logprob']:.4f} beam_score={hyp['beam_score']:.4f} {hyp['text']}")
    first_tokens = hyps[0]["tokens"][:8]
    print("1-best leading tokens:", [(t["tok"], round(t["logprob"], 3)) for t in first_tokens])

    # score_text round trip: the model's own output must outscore a corrupted variant
    own = hyps[0]["text"]
    words = own.split()
    victim = max(range(len(words)), key=lambda i: len(words[i]))
    original = words[victim]
    words[victim] = "flurbish"
    corrupted = " ".join(words)

    own_score = engine.score_text(audio, own)
    bad_score = engine.score_text(audio, corrupted)
    print(f"score_text own:       mean_all={own_score['mean_all']:.4f}")
    print(f"score_text corrupted: mean_all={bad_score['mean_all']:.4f} ({original} -> flurbish)")
    if own_score["mean_all"] <= bad_score["mean_all"]:
        raise RuntimeError("score_text round trip failed: corrupted text outscored the model's own output")
    print("round trip passed")
