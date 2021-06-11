"""Gooya script"""

__author__ = 'Vardë'

from typing import List, Set, Tuple

import lvsfunc as lvf
import vapoursynth as vs
import vardefunc as vdf
from G41Fun import DetailSharpen, MaskedDHA, SMDegrain
from muvsfunc import SSIM_downsample
from vardautomation import (ENGLISH, FSRCNNX_56_16_4_1, FileInfo,
                            MatroskaXMLChapters, OGMChapters, PresetAAC,
                            PresetBD)
from vsgan import VSGAN
from vsutil import depth, get_y, insert_clip, iterate, join, split

core = vs.core

JPDVD = FileInfo(
    "../GOOYA_S2_DVDISO/[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]/VIDEO_TS/VTS_01_1.dgi", 24, 34550,
    idx=lambda x: core.dgdecodenv.DGSource(x, fieldop=1),
    preset=[PresetBD, PresetAAC]
)

JPDVD.chapter = "../GOOYA_S2_DVDISO/[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]/VTS_01_0__VTS_01_1_1.txt"


OPSTART, OPEND = 2350, 4506


CLIP_JPBD: List[vs.VideoNode] = [
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00002.m2ts'),

    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい？～天才たちの恋愛頭脳戦～2/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい？～天才たちの恋愛頭脳戦～2/BDMV/BDMV/STREAM/00002.m2ts'),

    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~3/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~3/BDMV/BDMV/STREAM/00002.m2ts'),

    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~4/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~4/BDMV/BDMV/STREAM/00002.m2ts'),

    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~5/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~5/BDMV/BDMV/STREAM/00002.m2ts'),

    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~6/BDMV/BDMV/STREAM/00001.m2ts'),
    lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~6/BDMV/BDMV/STREAM/00002.m2ts'),
]

CLIP_JPBD_NCOP = lvf.misc.source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00004.m2ts')




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
        out = src



        full = vdf.scale.to_444(out, znedi=True)
        if isinstance(full, list):
            raise TypeError
        out = full



        unfuck_a = SMDegrain(out, thSAD=325).knlm.KNLMeansCL(
            h=0.8, a=2, d=3, device_type='gpu', device_id=0, channels='YUV'
        )
        unfuck_b = SMDegrain(out, thSAD=170).knlm.KNLMeansCL(
            h=0.4, a=1, d=2, device_type='gpu', device_id=0, channels='YUV'
        )
        unfuck_c = SMDegrain(out, thSAD=500).knlm.KNLMeansCL(
            h=2.2, a=4, d=3, device_type='gpu', device_id=0, channels='YUV'
        )
        unfuck = out
        unfuck = lvf.misc.replace_ranges(unfuck, unfuck_a, [(6553, 6582)])
        unfuck = lvf.misc.replace_ranges(unfuck, unfuck_b, [(16893, 16945), (17570, 17650), (22298, 22405)])
        unfuck = lvf.misc.replace_ranges(unfuck, unfuck_c, [(17039, 17172)])
        out = unfuck


        gan = self.gan_upscale(out)
        w2x = self.waifu2x_upscale(out)
        ai_upscale = lvf.misc.replace_ranges(gan, w2x, [(13750, 13833), (14548, 14805)])

        planes = split(ai_upscale)

        # planes[0] = self.fsrcnnx_upscale(planes[0], 1920, 1080, FSRCNNX_56_16_4_1)
        planes[0] = self.nnedi_upscale(planes[0], 1920, 1080)

        planes[1], planes[2] = [
            p.resize.Bicubic(960, 540, src_left=-0.5 * p.height/1080,
                             filter_param_a=-.5, filter_param_b=.25)
            for p in planes[1:]
        ]

        merged = join(planes)
        out = depth(merged, 16)




        dehalo = MaskedDHA(out, 1.4, 1.4, 0.05, maskpull=48, maskpush=192)
        out = dehalo




        deband_mask = self.deband_mask(out, kirsch_brz=(2000, 3500, 3500), rng_brz=(2000, 2000, 2000))

        deband = vdf.deband.dumb3kdb(out, 16, 42)
        deband_a = vdf.placebo.deband(out, 20, 5.6, iterations=2, grain=1.0)
        deband = lvf.misc.replace_ranges(
            deband, deband_a,
            [(11191, 11226), (12127, 12246), (12727, 12858), (13834, 13875),
             (14128, 14163), (16256, 16285)]
        )

        deband = core.std.MaskedMerge(deband, out, deband_mask)
        out = deband



        opening = insert_clip(out, self.make_opening(), OPSTART)
        opening = lvf.misc.replace_ranges(
            opening, out, [(OPSTART+746, OPSTART+799), (OPSTART+920, OPSTART+999)]
        )
        out = opening




        thrs = [x << 8 for x in (32, 80, 128, 176)]
        strengths = [(0.35, 0.2), (0.25, 0.15), (0.15, 0.0), (0.0, 0.0)]
        sizes = (1.2, 1.1, 1, 1)
        sharps = (70, 60, 50, 50)
        grainers = [vdf.noise.AddGrain(seed=333, constant=False),
                    vdf.noise.AddGrain(seed=333, constant=False),
                    vdf.noise.AddGrain(seed=333, constant=True)]
        pref = iterate(get_y(out), core.std.Maximum, 2).std.Convolution([1]*9)
        grain = vdf.noise.Graigasm(thrs, strengths, sizes, sharps, grainers=grainers).graining(out, prefilter=pref)
        out = grain


        decs = vdf.noise.decsiz(out, min_in=thrs[-2], max_in=thrs[-1])
        out = decs




        return src.resize.Spline16(1920, 1080), out

    @staticmethod
    def gan_upscale(clip: vs.VideoNode) -> vs.VideoNode:
        if clip.format is None:
            raise vdf.util.FormatError('No variable format')

        if clip.format.subsampling_w != 0 and clip.format.subsampling_h != 0:
            raise vdf.util.FormatError('Should be in 444')

        clip = clip.resize.Point(format=vs.RGB24, dither_type='error_diffusion')

        vsgan = VSGAN('cuda')
        vsgan.load_model(r'..\training\BasicSR\experiments\Kaguya_template_001\models\95000_G.pth')

        return vsgan.run(clip).resize.Point(format=vs.YUV444PS, matrix=1)

    @staticmethod
    def waifu2x_upscale(clip: vs.VideoNode) -> vs.VideoNode:
        if clip.format is None:
            raise vdf.util.FormatError('No variable format')

        if clip.format.subsampling_w != 0 and clip.format.subsampling_h != 0:
            raise vdf.util.FormatError('Should be in 444')

        clip = clip.resize.Point(format=vs.RGBS)

        return core.w2xnvk.Waifu2x(clip, noise=0, model=2).resize.Point(format=vs.YUV444PS, matrix=1)


    def nnedi_upscale(self, clip: vs.VideoNode, width: int, height: int) -> vs.VideoNode:
        if clip.format.num_planes != 1:
            raise vdf.util.FormatError('Only luma, get out with your chroma')

        if width < clip.width or height < clip.height:
            raise ValueError('Only upscaling allowed')

        clip = vdf.scale.nnedi3_upscale(clip, correct_shift=False, pscrn=1)
        clip = self.ssim_dowsample(clip, width, height)
        return clip

    def fsrcnnx_upscale(self, clip: vs.VideoNode, width: int, height: int, shader_file) -> vs.VideoNode:
        if clip.format.num_planes != 1:
            raise vdf.util.FormatError('Only luma, get out with your chroma')

        if width < clip.width or height < clip.height:
            raise ValueError('Only upscaling allowed')


        clip = vdf.scale.fsrcnnx_upscale(
            clip, width, height, shader_file, self.ssim_dowsample,
            profile='zastin', sharpener=lambda x: DetailSharpen(x, sstr=1.65, mode=0, med=True)
        )

        return clip

    def make_opening(self) -> vs.VideoNode:
        ops_bd = [
            CLIP_JPBD[1][:2158],

            CLIP_JPBD[2][:2158],
            CLIP_JPBD[3][2230:4387],

            CLIP_JPBD[4][:2158],
            CLIP_JPBD[5][:2158],

            CLIP_JPBD[6][2662:4819],
            CLIP_JPBD[7][:2158],

            CLIP_JPBD[8][408:2566],
            CLIP_JPBD[9][1296:3453]
        ]
        ops_bd = [depth(c, 32)[:2157] for c in ops_bd]


        mean = core.average.Mean(ops_bd)
        out = mean


        denoise = lvf.denoise.bm3d(out, [1.0, 0.75, 0.75], 1,
                                   basic_args=dict(profile='fast'),
                                   final_args=dict(profile='fast'))
        denoise = depth(denoise, 16)
        out = denoise

        dehalo = MaskedDHA(out, 1.3, 1.3, 0.05, maskpull=48, maskpush=192)
        out = dehalo


        aaa = lvf.aa.upscaled_sraa(out, 2, downscaler=lvf.kernels.Bicubic(b=-0.5, c=0.25).scale)
        aaa = core.rgvs.Repair(aaa, out, 13)
        out = aaa

        credits_mask = vdf.mask.diff_creditless_mask(
            out.std.BlankClip(length=out.num_frames + 1), mean, CLIP_JPBD_NCOP,
            start_frame=0, thr=25 << 8
        ).std.Convolution([1]*9)
        ref = denoise
        credit = core.std.MaskedMerge(out, ref, credits_mask[:out.num_frames])
        out = credit



        deband = vdf.deband.dumb3kdb(out, 16, 30)
        deband_mask = self.deband_mask(out, kirsch_brz=(2000, 3500, 3500), rng_brz=(2000, 2000, 2000))
        deband = core.std.MaskedMerge(deband, out, deband_mask)
        out = deband


        return out

    @staticmethod
    def deband_mask(clip: vs.VideoNode, kirsch_brz: Tuple[int, int, int], rng_brz: Tuple[int, int, int]) -> vs.VideoNode:
        prefilter = core.bilateral.Bilateral(clip, sigmaS=1.5, sigmaR=0.005)

        kirsch = vdf.mask.Kirsch().get_mask(prefilter).std.Binarize(kirsch_brz)
        rng = lvf.mask.range_mask(prefilter, 3, 2).std.Binarize(rng_brz)
        kirsch, rng = [c.resize.Bilinear(format=vs.YUV444P16) for c in [kirsch, rng]]

        mask = core.std.Expr(split(kirsch) + split(rng), vdf.util.max_expr(6))

        return mask.rgvs.RemoveGrain(22).rgvs.RemoveGrain(11)

    @staticmethod
    def ssim_dowsample(c: vs.VideoNode, w: int, h: int) -> vs.VideoNode:
        return SSIM_downsample(
            c, w, h, smooth=((3 ** 2 - 1) / 12) ** 0.5,
            sigmoid=True, filter_param_a=0, filter_param_b=0
        )


class ChapterStuff:
    @staticmethod
    def main():
        fps = JPDVD.clip_cut.fps


        assert JPDVD.chapter
        chapters = OGMChapters(JPDVD.chapter).ogm_to_chapters(fps, ENGLISH)

        path_chapters_xml = 'kaguya_ova_chapters.xml'
        chapters_xml = MatroskaXMLChapters(path_chapters_xml)
        chapters_xml.create(chapters, fps)

        chapters_xml.set_names(['Intro', 'OP', 'Part A', 'Part B', 'Part C', 'ED'])

        JPDVD.chapter = path_chapters_xml



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
