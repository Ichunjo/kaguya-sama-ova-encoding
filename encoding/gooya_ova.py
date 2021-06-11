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
    filtered = Filtering().main()
    filtered[0].set_output(0)
    filtered[1].set_output(1)
    # print(filtered)

    # Filtering().make_opening().set_output(1)

    # core.std.MakeDiff(filtered.resize.Bicubic(format=vs.YUV420P8)[2350:4507], CLIP_JPBD[1][:2157]).set_output(1)

    # dvd = filtered[0][2350:4507].resize.Spline16(format=vs.YUV420P8)
    
    # dvd += dvd[-1]
    # dvd.set_output(0)

    # filtered[1].set_output(1)
    """
    CLIP_JPBD[1] = CLIP_JPBD[1][:2158]
    CLIP_JPBD[2] = CLIP_JPBD[2][:2158]
    CLIP_JPBD[3] = CLIP_JPBD[3][2230:4387]
    CLIP_JPBD[4] = CLIP_JPBD[4][:2158]
    CLIP_JPBD[5] = CLIP_JPBD[5][:2158]
    CLIP_JPBD.pop(0)

    # diffs = []

    for i, ncop in enumerate(CLIP_JPBD, start=1):
        # print(ncop)
        core.std.MakeDiff(dvd, ncop).set_output(i)
        # diffs.append(
        #     core.std.MakeDiff(dvd, ncop)
        # )
    """
    # CLIP_JPBD[11].set_output(0)

    # test = CLIP_JPBD[6][2662:4819]

    # test = depth(test, 16)
    # test.set_output(0)


    # denoise = lvf.denoise.bm3d(test, [1.0, 0.75, 0.75], 1,
    #                            basic_args=dict(profile='fast'),
    #                            final_args=dict(profile='fast'))
    # out = denoise

    # dehalo = MaskedDHA(out, 1.4, 1.4, 0.05, maskpull=48, maskpush=192)
    # out = dehalo


    # aaa = lvf.aa.upscaled_sraa(out, 2, downscaler=lvf.kernels.Bicubic(b=-0.5, c=0.25).scale)
    # aaa = core.rgvs.Repair(aaa, out, 13)
    # out = aaa

    # out.set_output(1)


    # CLIP_JPBD[7] = CLIP_JPBD[6][:2158]
    # CLIP_JPBD[8] = CLIP_JPBD[7][408:2566]
    # CLIP_JPBD[9] = CLIP_JPBD[8][1296:3453]

    # CLIP_JPBD[5].set_output(2)

    # JPDVD.clip_cut.resize.Spline36(1920, 1080).set_output(0)



# CLIP_JPBD[6].set_output(0)
# lvf.comparison.stack_compare(*OP_BD, make_diff=False).set_output(0)


# ref = OP_BD[0]
# ref.set_output(0)
# for i, op in enumerate(OP_BD[1:], start=1):
#     try:
#         diff = lvf.comparison.diff(ref, op)
#     except ValueError:
#         pass
#     else:
#         diff.text.Text(i).set_output(i)
