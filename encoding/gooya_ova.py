"""Gooya script"""

__author__ = 'Vardë'

from typing import Set, Tuple
import vapoursynth as vs
import vardefunc as vdf
from vardautomation import FileInfo, PresetAAC, PresetBD
from vsutil import depth
from vsgan import VSGAN

core = vs.core

JPDVD = FileInfo(
    r"..\GOOYA_S2_DVDISO\[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]\VIDEO_TS\VTS_01_1.dgi", 24, 34550,
    idx=lambda x: core.dgdecodenv.DGSource(x, fieldop=1),
    preset=[PresetBD, PresetAAC]
)



PROPS_DVD: Set[Tuple[str, int]] = {
    ('_ChromaLocation', 0),
    ('_Matrix', 5),
    ('_Transfer', 5),
    ('_Primaries', 5),
    ('_FieldBased', 0)
}



class Filtering:
    def main(self) -> vs.VideoNode:
        src = JPDVD.clip_cut

        for prop, val in PROPS_DVD:
            src = src.std.SetFrameProp(prop, intval=val)

        src = depth(src, 16)


        full = vdf.scale.to_444(src, znedi=True)
        if isinstance(full, list):
            raise TypeError

        gan = self.upscale_gan(full)
        nnedi = vdf.scale.nnedi3_upscale(gan).resize.Bicubic(1920, 1080)

        return src.resize.Bicubic(1920, 1080), nnedi

    @staticmethod
    def upscale_gan(clip: vs.VideoNode) -> vs.VideoNode:
        clip = core.resize.Point(clip, format=vs.RGB24, dither_type='error_diffusion')

        vsgan = VSGAN('cuda')
        vsgan.load_model(r'..\training\BasicSR\experiments\Kaguya_template_001\models\95000_G.pth')

        return vsgan.run(clip).resize.Point(format=vs.YUV444PS, matrix=1)



if __name__ == '__main__':
    pass
else:
    clips = Filtering().main()
    a, b = clips
    a.set_output(0)
    b.set_output(1)
