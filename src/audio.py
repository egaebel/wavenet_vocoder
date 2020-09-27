from hparams import hparams as wavenet_hparams
from nnmnkwii import preprocessing as P
from scipy.io import wavfile
from torch_stft import STFT

import librosa
import librosa.filters
import lws
import numpy as np
import torch


_torch_mel_basis = None
_torch_stft_instance = None


def low_cut_filter(x, fs, cutoff=70):
    """APPLY LOW CUT FILTER.

    https://github.com/kan-bayashi/PytorchWaveNetVocoder

    Args:
        x (ndarray): Waveform sequence.
        fs (int): Sampling frequency.
        cutoff (float): Cutoff frequency of low cut filter.
    Return:
        ndarray: Low cut filtered waveform sequence.
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    from scipy.signal import firwin, lfilter

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2 ** 15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    if sr != wavenet_hparams.sample_rate:
        x = librosa.resample(x, sr, wavenet_hparams.sample_rate)
    x = np.clip(x, -1.0, 1.0)
    return x


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, wavenet_hparams.sample_rate, wav.astype(np.int16))


def trim(quantized):
    start, end = start_and_end_indices(quantized, wavenet_hparams.silence_threshold)
    return quantized[start:end]


def preemphasis(x, coef=0.85):
    return P.preemphasis(x, coef)


def inv_preemphasis(x, coef=0.85):
    return P.inv_preemphasis(x, coef)


def adjust_time_resolution(quantized, mel):
    """Adjust time resolution by repeating features

    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)

    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, wavenet_hparams.silence_threshold)

    return quantized[start:end], mel[start:end, :]


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def logmelspectrogram(y, pad_mode="reflect", use_lws=False):
    global _mel_basis
    global _torch_mel_basis
    if use_lws:
        return melspectrogram_lws(y)
    """Same log-melspectrogram computation as espnet
    https://github.com/espnet/espnet
    from espnet.transform.spectrogram import logmelspectrogram
    """
    D = _stft(y, pad_mode=pad_mode)
    if wavenet_hparams.use_torch_stft:

        if _mel_basis is None or _torch_mel_basis is None:
            _mel_basis = _build_mel_basis()
            _torch_mel_basis = torch.from_numpy(_mel_basis).float()

        # print("D[0]: '%s'" % str(D[0]))
        magnitudes = D[0].data
        mel_output = torch.matmul(_torch_mel_basis, magnitudes)
        return mel_output.squeeze().cpu().data.numpy()
    else:
        S = _linear_to_mel(np.abs(D))
        S = np.log10(np.maximum(S, 1e-10))
    return S


def get_hop_size():
    hop_size = wavenet_hparams.hop_size
    if hop_size is None:
        print(
            "hop hop_size is None, computing hop_size from wavenet_hparams.frame_shift_ms: '%s'"
            % str(wavenet_hparams.frame_shift_ms),
            flush=True,
        )
        assert wavenet_hparams.frame_shift_ms is not None
        hop_size = int(
            wavenet_hparams.frame_shift_ms / 1000 * wavenet_hparams.sample_rate
        )
    return hop_size


def get_win_length():
    win_length = wavenet_hparams.win_length
    if win_length < 0:
        assert wavenet_hparams.win_length_ms > 0
        win_length = int(
            wavenet_hparams.win_length_ms / 1000 * wavenet_hparams.sample_rate
        )
    return win_length


def _stft(y, pad_mode="constant"):
    global _torch_stft_instance
    if wavenet_hparams.use_torch_stft:
        y_torch = torch.FloatTensor(y)
        y_torch = y_torch.unsqueeze(0)
        if _torch_stft_instance is None:
            _torch_stft_instance = STFT(
                filter_length=wavenet_hparams.fft_size,
                hop_length=get_hop_size(),
                win_length=get_win_length(),
                window=wavenet_hparams.window,
            ).to("cpu")
        return _torch_stft_instance.transform(y_torch)
    # use constant padding (defaults to zeros) instead of reflection padding
    return librosa.stft(
        y=y,
        n_fft=wavenet_hparams.fft_size,
        hop_length=get_hop_size(),
        win_length=get_win_length(),
        window=wavenet_hparams.window,
        pad_mode=pad_mode,
    )


def pad_lr(x, fsize, fshift):
    return (0, fsize)


# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    if wavenet_hparams.fmax is not None:
        assert wavenet_hparams.fmax <= wavenet_hparams.sample_rate // 2
    return librosa.filters.mel(
        wavenet_hparams.sample_rate,
        wavenet_hparams.fft_size,
        fmin=wavenet_hparams.fmin,
        fmax=wavenet_hparams.fmax,
        n_mels=wavenet_hparams.num_mels,
    )


def _amp_to_db(x):
    min_level = np.exp(wavenet_hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip(
        (S - wavenet_hparams.log_scale_min) / -wavenet_hparams.log_scale_min, 0, 1
    )


def _denormalize(S):
    return (
        np.clip(S, 0, 1) * -wavenet_hparams.log_scale_min
    ) + wavenet_hparams.log_scale_min


# lws alternatives
def melspectrogram_lws(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - wavenet_hparams.ref_level_db
    if not wavenet_hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - wavenet_hparams.log_scale_min >= 0
    return _normalize(S)


def _lws_processor():
    return lws.lws(wavenet_hparams.fft_size, get_hop_size(), mode="speech")
