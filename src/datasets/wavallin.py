from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from hparams import hparams
from nnmnkwii import preprocessing as P
from os.path import exists, basename, splitext
from os.path import join
from train import assert_ready_for_upsampling
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw

import audio
import librosa
import numpy as np
import os
import subprocess


WAV_FILES_DIR = "converted-wav-files"


def _convert_mp3_to_wav(input_file_path, output_file_path, sample_rate=None):
    ffmpeg_command = [
        "ffmpeg",
        # Overwrite file if it already exists.
        "-y",
        "-i",
        "%s" % input_file_path,
        # Output one audio channel.
        "-ac",
        "1",
    ]
    if sample_rate is not None:
        ffmpeg_command.append("-ar")
        ffmpeg_command.append("%s" % str(sample_rate))
    ffmpeg_command.append("%s" % output_file_path,)
    subprocess.check_output(ffmpeg_command)


def _try_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


def build_from_path(in_dir, out_dir, num_workers=1, sample_rate=None, tqdm=lambda x: x):
    _try_mkdir(WAV_FILES_DIR)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    src_files = sorted(
        list(glob(join(in_dir, "*.wav"))) + list(glob(join(in_dir, "*.mp3")))
    )
    print(
        "Building from in_dir: '%s' with '%d' src files....." % (in_dir, len(src_files))
    )
    for audio_file_path in src_files:

        audio_file_path_without_ext, audio_file_ext = os.path.splitext(audio_file_path)
        audio_file_name_without_ext = os.path.basename(audio_file_path_without_ext)

        if audio_file_ext == ".mp3":
            wav_audio_file_name = audio_file_name_without_ext + ".wav"
            wav_file_path = os.path.join(WAV_FILES_DIR, wav_audio_file_name)

            try:
                if not os.path.exists(wav_file_path):
                    _convert_mp3_to_wav(
                        audio_file_path, wav_file_path, sample_rate=sample_rate
                    )
            except subprocess.CalledProcessError as e:
                print(
                    "CalledProcessError on converting from: ||%s|| to: ||%s||\n\nException: %s\n\nSkipping....."
                    % (audio_file_path, wav_file_path, str(e))
                )
                continue
        elif audio_file_ext == ".wav":
            pass
        else:
            raise ValueError(
                "Received unexpected audio file extension: ||%s||" % audio_file_ext
            )

        futures.append(
            executor.submit(
                partial(_process_utterance, out_dir, index, wav_file_path, "dummy")
            )
        )
        index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, trim_silence=False):
    # Load the audio to a numpy array:

    wav = audio.load_wav(wav_path)

    # Trim begin/end silences
    # NOTE: the threshold was chosen for clean signals
    # TODO: Remove, get this out of here.
    if trim_silence:
        wav, _ = librosa.effects.trim(wav, top_db=60, frame_length=2048, hop_length=512)

    if hparams.highpass_cutoff > 0.0:
        wav = audio.low_cut_filter(wav, hparams.sample_rate, hparams.highpass_cutoff)

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # Trim silences in mul-aw quantized domain
        silence_threshold = 0
        if silence_threshold > 0:
            # [0, quantize_channels)
            out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
            start, end = audio.start_and_end_indices(out, silence_threshold)
            wav = wav[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        constant_values = P.mulaw(0.0, hparams.quantize_channels - 1)
        out_dtype = np.float32
    else:
        # [-1, 1]
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.logmelspectrogram(wav).astype(np.float32).T

    if hparams.global_gain_scale > 0:
        wav *= hparams.global_gain_scale

    # Time domain preprocessing
    if hparams.preprocess is not None and hparams.preprocess not in ["", "none"]:
        f = getattr(audio, hparams.preprocess)
        wav = f(wav)

    # Clip
    if np.abs(wav).max() > 1.0:
        print("""Warning: abs max value exceeds 1.0: {}""".format(np.abs(wav).max()))
        # ignore this sample
        return ("dummy", "dummy", -1, "dummy")

    wav = np.clip(wav, -1.0, 1.0)

    # Set waveform target (out)
    if is_mulaw_quantize(hparams.input_type):
        out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        out = P.mulaw(wav, hparams.quantize_channels - 1)
    else:
        out = wav

    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.pad_lr(out, hparams.fft_size, audio.get_hop_size())
    if l > 0 or r > 0:
        out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[: N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    assert_ready_for_upsampling(out, mel_spectrogram, cin_pad=0, debug=True)

    # Write the spectrograms to disk:
    name = splitext(basename(wav_path))[0]
    audio_filename = "%s-wave.npy" % (name)
    mel_filename = "%s-feats.npy" % (name)
    np.save(
        os.path.join(out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False
    )
    np.save(
        os.path.join(out_dir, mel_filename),
        mel_spectrogram.astype(np.float32),
        allow_pickle=False,
    )

    # Return a tuple describing this training example:
    return (audio_filename, mel_filename, N, text)
