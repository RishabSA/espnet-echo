import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput

sample_rate = 16000
# sequences per teacher-forced rescoring forward; vocab-wide logits are ~40 MB per
# sequence of length 200, so this bounds peak memory regardless of batch x n-best
rescore_batch = 8


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WhisperEngine:
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else pick_device()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=dtype
        ).to(self.device)
        self.model.eval()
        self.model_id = model_id
        self.revision = revision

    def _start_ids(self, language: str, prompt: str | None) -> tuple[list[int], int]:
        tokenizer = self.processor.tokenizer
        start_ids = [
            tokenizer.convert_tokens_to_ids(t)
            for t in ("<|startoftranscript|>", f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>")
        ]
        prefix_len = 0
        if prompt is not None:
            prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors="pt").flatten().tolist()
            start_ids = prompt_ids + start_ids
            prefix_len = len(prompt_ids)
        return start_ids, prefix_len

    def _token_logprobs(self, features: torch.Tensor, sequences: torch.Tensor, n_return: int) -> torch.Tensor:
        # per-token logprobs from a batched teacher-forced forward instead of
        # compute_transition_scores: version-proof, and it puts pass-1 confidences on
        # exactly the same raw-model scale score_text uses, which is what delta_null
        # comparisons assume; the forward sees the full sequence including any prompt
        # prefix, so logprobs are conditioned exactly as the decode was
        with torch.inference_mode():
            encoder_states = self.model.get_encoder()(features).last_hidden_state
            encoder_states = encoder_states.repeat_interleave(n_return, dim=0)

            pieces = []
            for lo in range(0, sequences.shape[0], rescore_batch):
                hi = min(lo + rescore_batch, sequences.shape[0])
                logits = self.model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=encoder_states[lo:hi]),
                    decoder_input_ids=sequences[lo:hi],
                ).logits
                logprob_dist = torch.log_softmax(logits.float(), dim=-1)
                target = sequences[lo:hi, 1:]  # shape: (b, seq_len - 1)
                pieces.append(
                    logprob_dist[:, :-1].gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
                )
        return torch.cat(pieces, dim=0)  # shape: (n_sequences, seq_len - 1)

    def n_best_decode_batch(
        self,
        audios: list[torch.Tensor],
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
        prompt: str | None = None,
    ) -> list[list[dict]]:
        features = self.processor(
            [a.numpy() for a in audios], sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.model.dtype)
        tokenizer = self.processor.tokenizer

        # bypass Whisper's custom generate: in transformers 5 it collapses
        # num_return_sequences to num_beams copies of the top beam (verified
        # 2026-08-09), while the vanilla path returns genuinely distinct beams,
        # which is the whole point of retaining n-best
        start_ids, prefix_len = self._start_ids(language, prompt)
        decoder_input_ids = torch.tensor([start_ids], device=self.device).repeat(len(audios), 1)

        with torch.inference_mode():
            out = GenerationMixin.generate(
                self.model,
                features,
                decoder_input_ids=decoder_input_ids,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        token_lp_full = self._token_logprobs(features, out.sequences, num_return_sequences)
        # drop any prompt prefix so position 0 is <|startoftranscript|> and prompt
        # words can never leak into texts or token lists
        sequences = out.sequences[:, prefix_len:]
        token_lp = token_lp_full[:, prefix_len:]
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        # length-penalised beam scores exist only under beam search; greedy falls back
        # to the length-normalised sum, the same objective at penalty 1.0
        seq_scores = getattr(out, "sequences_scores", None)

        results = []
        for b in range(len(audios)):
            hyps = []
            for rank in range(num_return_sequences):
                i = b * num_return_sequences + rank
                ids = sequences[i, 1:].tolist()
                toks = tokenizer.convert_ids_to_tokens(ids)
                logprobs = token_lp[i].tolist()

                # keep text tokens only: specials carry no acoustic evidence and
                # padding past end-of-text would corrupt the sums
                tokens = [
                    {"id": tok_id, "tok": tok, "logprob": lp}
                    for tok_id, tok, lp in zip(ids, toks, logprobs, strict=True)
                    if not (tok.startswith("<|") and tok.endswith("|>"))
                ]
                sum_logprob = sum(t["logprob"] for t in tokens)
                avg_logprob = sum_logprob / len(tokens) if tokens else 0.0
                hyps.append(
                    {
                        "rank": rank,
                        "text": texts[i].strip(),
                        "tokens": tokens,
                        "sum_logprob": sum_logprob,
                        "avg_logprob": avg_logprob,
                        "beam_score": seq_scores[i].item() if seq_scores is not None else avg_logprob,
                    }
                )
            results.append(hyps)
        return results

    def n_best_decode(
        self,
        audio: torch.Tensor,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
        prompt: str | None = None,
    ) -> list[dict]:
        return self.n_best_decode_batch(
            [audio],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            language=language,
            prompt=prompt,
        )[0]

    def score_text(
        self,
        audio: torch.Tensor,
        text: str,
        focus: tuple[int, int] | None = None,
        language: str = "en",
    ) -> dict:
        # teacher-forced scoring of arbitrary text; focus is a character range in text
        # whose token logprobs are summed separately (spec section 6.8, canonicalizer b)
        inputs = self.processor(audio.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        tokenizer = self.processor.tokenizer

        # leading space matches Whisper's own output convention for mid-transcript words
        prefixed = " " + text.strip()
        enc = tokenizer(prefixed, add_special_tokens=False, return_offsets_mapping=True)
        text_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        special, _ = self._start_ids(language, None)
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        decoder_ids = torch.tensor([special + text_ids + [eot]], device=self.device)

        with torch.inference_mode():
            logits = self.model(features, decoder_input_ids=decoder_ids).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)  # shape: (1, seq, vocab)
        # next-token logprob for decoder position t sits at logits index t - 1
        target = decoder_ids[0, 1:]  # shape: (seq - 1,)
        token_lp = logprobs[0, :-1].gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)

        # text token j occupies decoder position len(special) + j, so target index
        # len(special) + j - 1
        text_lp = token_lp[len(special) - 1 : len(special) - 1 + len(text_ids)]
        result = {
            "sum_all": text_lp.sum().item(),
            "mean_all": text_lp.mean().item(),
            "n_tokens": len(text_ids),
        }

        if focus is not None:
            # offsets are relative to the prefixed string, shift the caller's range by 1
            lo, hi = focus[0] + 1, focus[1] + 1
            idx = [j for j, (s, e) in enumerate(offsets) if s < hi and e > lo]
            if not idx:
                raise ValueError(
                    f"focus range {focus} selects no tokens in text of length {len(text)}"
                )
            focus_lp = text_lp[idx]
            result.update(
                {
                    "sum_focus": focus_lp.sum().item(),
                    "mean_focus": focus_lp.mean().item(),
                    "n_focus_tokens": len(idx),
                }
            )
        return result

    def biased_decode(
        self,
        audio: torch.Tensor,
        prompt: str,
        num_beams: int = 8,
        num_return_sequences: int = 8,
        language: str = "en",
    ) -> list[dict]:
        return self.n_best_decode(
            audio,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            language=language,
            prompt=prompt,
        )
