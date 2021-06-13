"""Gooya script"""

__author__ = 'Vardë'

from pathlib import Path

import vapoursynth as vs
from vardautomation import FileInfo, X265Encoder
from vsutil import insert_clip

core = vs.core


ENCODER = X265Encoder('x265', Path('x265_settings_576p'))




file = FileInfo("gooya_ova_lossless_1part.mkv")


clip = file.clip
fix1 = core.ffms2.Source('gooya_ova_lossless_13750_13833+1.mkv')
fix2 = core.ffms2.Source('gooya_ova_lossless_14548_14805+1.mkv')

clip = insert_clip(clip, fix1, 13750)
clip = insert_clip(clip, fix2, 14548)


clip.set_output(0)

clip = core.resize.Bicubic(
    clip, 1024, 576, vs.YUV444P10,
    dither_type='error_diffusion',
    filter_param_a=1/3, filter_param_b=1/3,
    filter_param_a_uv=0, filter_param_b_uv=0.5
)


ENCODER.run_enc(clip, file)
