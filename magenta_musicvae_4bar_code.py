import glob

BASE_DIR = "C:/magenta_tutorial/checkpoints"

import ctypes.util
orig_ctypes_util_find_library = ctypes.util.find_library
def proxy_find_library(lib):
  if lib == 'fluidsynth':
    return 'libfluidsynth.so.1'
  else:
    return orig_ctypes_util_find_library(lib)
ctypes.util.find_library = proxy_find_library


print('Importing libraries and defining some helper functions...')
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import numpy as np
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Necessary until pyfluidsynth is updated (>1.2.5).
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def interpolate(model, start_seq, end_seq, num_steps, max_length=32,
                assert_same_length=True, temperature=0.5,
                individual_duration=4.0):
  """Interpolates between a start and end sequence."""
  note_sequences = model.interpolate(
      start_seq, end_seq,num_steps=num_steps, length=max_length,
      temperature=temperature,
      assert_same_length=assert_same_length)

  print('Mean Sequence')
  print('Start -> End Interpolation')
  interp_seq = mm.sequences_lib.concatenate_sequences(
      note_sequences, [individual_duration] * len(note_sequences))
  mm.plot_sequence(interp_seq)
  return interp_seq if num_steps > 3 else note_sequences[num_steps // 2]

def download(note_sequence, filename):
  mm.sequence_proto_to_midi_file(note_sequence, filename)

print('Done')


drums_models = {}
# One-hot encoded.
drums_config = configs.CONFIG_MAP['groovae_4bar']
drums_models['groovae_4bar'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/groovae_4bar.tar')

drums_config = configs.CONFIG_MAP['groovae_2bar_humanize']
drums_models['groovae_2bar_humanize'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/groovae_2bar_humanize.tar')

drums_config = configs.CONFIG_MAP['groovae_2bar_tap_fixed_velocity']
drums_models['groovae_2bar_tap_fixed_velocity'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/groovae_2bar_tap_fixed_velocity.tar')

drums_config = configs.CONFIG_MAP['groovae_2bar_add_closed_hh']
drums_models['groovae_2bar_add_closed_hh'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/groovae_2bar_add_closed_hh.tar')

drums_config = configs.CONFIG_MAP['cat-drums_2bar_small']
drums_models['cat-drums_2bar_small.lokl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/cat-drums_2bar_small.lokl.tar')
drums_models['cat-drums_2bar_small.hikl'] = TrainedModel(drums_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/cat-drums_2bar_small.hikl.tar')


# Multi-label NADE.
drums_nade_full_config = configs.CONFIG_MAP['nade-drums_2bar_full']
drums_models['nade-drums_2bar_full'] = TrainedModel(drums_nade_full_config, batch_size=4, checkpoint_dir_or_path=BASE_DIR + '/nade-drums_2bar_full.tar')



drums_sample_model = "groovae_4bar"
temperature = 0.5
drums_samples = drums_models[drums_sample_model].sample(n=4, length=32, temperature=temperature)



for i, ns in enumerate(drums_samples):
  download(ns, 'output/%s_sample_%d.mid' % (drums_sample_model, i))