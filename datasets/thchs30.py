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

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  # male voice (do not use) A5 A8 A9 A33 A35 B6 B8 B21 B34 C6 C8 D8
  # too silent (do not use) A36 B33 C14 D32

  # based upon https://github.com/xiaofengShi/Tacotron-Chinese
  trn_files = []

  trn_files += glob.glob(os.path.join(in_dir, 'A2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A14_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A19_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A23_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A32_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'A34_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'B2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B15_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B31_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'B32_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'C2_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C17_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C18_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C19_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C20_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C21_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C22_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C23_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C31_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'C32_*.trn'))

  trn_files += glob.glob(os.path.join(in_dir, 'D4_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D6_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D7_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D11_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D12_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D13_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D21_*.trn'))
  trn_files += glob.glob(os.path.join(in_dir, 'D31_*.trn'))

  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  # for file in os.listdir(in_dir):
  for file in trn_files:
    # if file[-4:] == ".trn":
      # with open(os.path.join(in_dir, file), 'r') as f:
    with open(file, 'r', encoding='utf-8') as f:
      text = (f.readlines())[1].strip()
      wav_path = file[:-4]
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
