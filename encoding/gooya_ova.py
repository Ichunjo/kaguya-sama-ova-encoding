"""Gooya script"""

__author__ = 'Vardë'

from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

import lvsfunc as lvf
import vapoursynth as vs
import vardefunc as vdf
from G41Fun import DetailSharpen, MaskedDHA, SMDegrain
from muvsfunc import SSIM_downsample
from vardautomation import (ENGLISH, FSRCNNX_56_16_4_1, AudioCutter,
                            AudioEncoder, BasicTool, EncodeGoBrr, FileInfo,
                            LosslessEncoder, MatroskaXMLChapters, OGMChapters,
                            PresetAAC, PresetBD, VideoEncoder, X265Encoder)
from vsgan import VSGAN
from vsutil import depth, get_y, insert_clip, iterate, join, split

core = vs.core

JPDVD = FileInfo(
    Path("../GOOYA_S2_DVDISO/[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]/VIDEO_TS/VTS_01_1.dgi").resolve(), 24, 34550,
    idx=lambda x: core.dgdecodenv.DGSource(x, fieldop=1),
    preset=[PresetBD, PresetAAC]
)
JPDVD.a_src = Path('../GOOYA_S2_DVDISO/[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]/VTS_01_1 Ta0 48K 16bit 2ch.wav').resolve()
JPDVD.chapter = Path("../GOOYA_S2_DVDISO/[DVDISO][210519][Kaguya-sama wa Kokurasetai S2][Vol.1 Fin]/VTS_01_0__VTS_01_1_1.txt").resolve()
JPDVD.do_lossless = True

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
        # ai_upscale = gan

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


        out = core.resize.Bicubic(
            out, 1024, 576, vs.YUV444P10,
            dither_type='error_diffusion',
            filter_param_a=1/3, filter_param_b=1/3,
            filter_param_a_uv=0, filter_param_b_uv=0.5
        )

        # return src.resize.Spline16(1920, 1080), out
        return depth(out, 10).std.Limiter(16 << 2, [235 << 2, 240 << 2], [0, 1, 2])

    @staticmethod
    def gan_upscale(clip: vs.VideoNode) -> vs.VideoNode:
        if clip.format is None:
            raise vdf.util.FormatError('No variable format')

        if clip.format.subsampling_w != 0 and clip.format.subsampling_h != 0:
            raise vdf.util.FormatError('Should be in 444')

        clip = clip.resize.Point(format=vs.RGB24, dither_type='error_diffusion')

        vsgan = VSGAN('cuda')
        vsgan.load_model('95000_G.pth')

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




class EncodeGoBrrr(EncodeGoBrr):
    def __init__(self, clip: vs.VideoNode, file: FileInfo, /,
                 v_encoder: VideoEncoder, v_lossless_encoder: Optional[LosslessEncoder],
                 a_extracters: Optional[Union[BasicTool, Sequence[BasicTool]]] = None,
                 a_cutters: Optional[Union[AudioCutter, Sequence[AudioCutter]]] = None,
                 a_encoders: Optional[Union[AudioEncoder, Sequence[AudioEncoder]]] = None) -> None:
        super().__init__(clip, file, v_encoder, v_lossless_encoder=v_lossless_encoder,
                         a_extracters=a_extracters, a_cutters=a_cutters, a_encoders=a_encoders)

    def run(self) -> None:
        self._parsing()
        self._encode()
        # self._audio_getter()
        # self.chapter()
        self.merge()

    def chapter(self) -> None:
        assert self.file.chapter
        assert self.file.frame_start

        fps = self.file.clip_cut.fps

        chapters = OGMChapters(self.file.chapter).to_chapters(fps, ENGLISH)

        path_chapters_xml = 'kaguya_ova_chapters.xml'
        chapters_xml = MatroskaXMLChapters(path_chapters_xml)
        chapters_xml.create(chapters, fps)
        chapters_xml.shift_times(0 - self.file.frame_start, fps)
        chapters_xml.set_names(['Intro', 'OP', 'Part A', 'Part B', 'Part C', 'ED'])

        self.file.chapter = path_chapters_xml


    def merge(self) -> None:
        assert self.file.chapter
        BasicTool('mkvmerge', [
            '-o', str(self.file.name_file_final),
            '--track-name', '0:HEVC BDRip by Vardë@Raws-Maji', '--language', '0:jpn', str(self.file.name_clip_output),
            '--tags', '0:tags_aac.xml', '--track-name', '0:AAC 2.0', '--language', '0:jpn', self.file.a_enc_cut.format(1),
            '--chapter-language', 'jpn', '--chapters', self.file.chapter
        ]).run()



def progress_update(v: int, e: int):
    return print(f"\rVapourSynth: {v}/{e} ~ {100 * v // e}% || Encoder: ", end="")


ENC_LOSS = LosslessEncoder(
    'ffmpeg',
    [
        '-i', '-',
        '-vcodec', 'ffv1',
        '-coder', '1', '-context', '0', '-g', '1', '-level', '3',
        '-threads', '16', '-slices', '24', '-slicecrc', '1', '-slicecrc', '1',
        str(JPDVD.name_clip_output_lossless)
    ],
    progress_update=progress_update
)

ENCODER = X265Encoder('x265', Path('x265_settings'), progress_update=progress_update)

A_CUTTER = AudioCutter(JPDVD, track=1)
A_ENCODER = AudioEncoder('qaac', Path('qaac_settings'), JPDVD, track=1)




if __name__ == '__main__':
    filtered = Filtering().main()

    brrrrr = EncodeGoBrrr(
        filtered, JPDVD,
        v_encoder=ENCODER, v_lossless_encoder=ENC_LOSS,
        a_cutters=A_CUTTER, a_encoders=A_ENCODER
    )
    brrrrr.run()
else:
    filtered = Filtering().main()
    # filtered.set_output(0)
    filtered[0].set_output(0)
    filtered[1].set_output(1)

