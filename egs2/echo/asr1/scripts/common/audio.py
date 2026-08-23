import soundfile as sf
import torch
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # shape: (frames, channels)
    audio = torch.from_numpy(wav).mean(dim=1)  # shape: (frames,)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio


def slice_audio(
    audio: torch.Tensor, start_s: float, end_s: float, sr: int = 16000, pad_s: float = 0.0
) -> torch.Tensor:
    start = max(0, int((start_s - pad_s) * sr))
    end = min(audio.shape[0], int((end_s + pad_s) * sr))
    if end <= start:
        raise ValueError(f"empty audio slice: start_s={start_s}, end_s={end_s}, pad_s={pad_s}")
    return audio[start:end]
