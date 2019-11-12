# -*- coding=utf-8 -*-
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
from util import audio


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  with open(os.path.join(in_dir, 'ProsodyLabeling/000001-010000.txt'), 'r', encoding='utf-8') as f:
    content = f.readlines()
    num = int(len(content)/2)
    index = 1
    for lineidx in range(num):
      wavidx_raw = content[lineidx*2].split()[0]
      wavidx = int(wavidx_raw)
      wavpath = os.path.join(in_dir, ('Wave/%s.wav' % wavidx_raw))
      text = content[lineidx*2+1].strip()
      print(wavpath, text)
      exit(0)
      futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
      index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'thchs30-spec-%05d.npy' % index
  mel_filename = 'thchs30-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
