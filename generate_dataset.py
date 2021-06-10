"""Generate a dataset for use in BasicSR"""


import random
from pathlib import Path
from typing import NamedTuple, Set, Tuple

import vapoursynth as vs
import vardefunc as vdf
from lvsfunc.misc import source
from lvsfunc.progress import (BarColumn, FPSColumn, Progress, TextColumn,
                              TimeRemainingColumn)
from lvsfunc.render import clip_async_render
from vsutil import depth

core = vs.core


# First import your clips
CLIP_DVD = source('GOOYA_S2_DVDISO/KAGUYA_S2_VOL1/VIDEO_TS/VTS_01_1.dgi', fieldop=1)
CLIP_BD_ = source('GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00001.m2ts') \
    + source('GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00002.m2ts')


# Since both DVD and BD doesn't have the props, we need to add them.
# If not, colours won't match.
PROPS_DVD: Set[Tuple[str, int]] = {
    ('_ChromaLocation', 0),
    ('_Matrix', 5),
    ('_Transfer', 5),
    ('_Primaries', 5)
}

PROPS_BD: Set[Tuple[str, int]] = {
    ('_Matrix', 1),
    ('_Transfer', 1),
    ('_Primaries', 1)
}


class ClipForDataset(NamedTuple):  # noqa: PLC0115
    clip: vs.VideoNode
    path: Path


class Dataset(NamedTuple):  # noqa: PLC0115
    hr: ClipForDataset
    lr: ClipForDataset



class PrepareDataset:  # noqa: PLC0115
    def prepare(self) -> Dataset:  # noqa: PLC0116
        # Make your adjustments to match the framerates and frames
        dvd = CLIP_DVD[:70513]
        bd_ = CLIP_BD_[:70513]


        # Force props
        for prop, val in PROPS_DVD:
            dvd = dvd.std.SetFrameProp(prop, intval=val)

        for prop, val in PROPS_BD:
            bd_ = bd_.std.SetFrameProp(prop, intval=val)


        # Prepare the HR and LR clip for dataset
        hr_ = self.prepare_hr(bd_)
        lr_ = self.prepare_lr(dvd)


        if hr_.format != lr_.format:
            raise vs.Error('Format of LR and HR do not match!')



        # We select 5 % of frames to make a reasonable size dataset
        if (length := dvd.num_frames) == dvd.num_frames:
            frames = sorted(random.sample(population=range(length), k=round(length*0.05)))
        else:
            raise IndexError("LR and HR don't have the same length!")

        # Select frames
        hr_ = core.std.Splice([hr_[f] for f in frames])
        lr_ = core.std.Splice([lr_[f] for f in frames])


        return Dataset(hr=ClipForDataset(hr_, Path('datasets/HR')),
                       lr=ClipForDataset(lr_, Path('datasets/LR')))


    @staticmethod
    def prepare_hr(clip: vs.VideoNode) -> vs.VideoNode:
        """Prepare HR clip in RGB24"""
        clip = depth(clip, 16)

        ups = vdf.scale.to_444(clip, znedi=True)
        if isinstance(ups, list):
            raise TypeError

        # The DVD is oddly cropped. We need to adjust the HR clip for a better training.
        ups = ups.std.Crop(top=6, bottom=7).resize.Bicubic(src_left=-0.25, src_top=0.5)

        return ups.resize.Bicubic(CLIP_DVD.width*2, CLIP_DVD.height*2,
                                  format=vs.RGB24, dither_type='error_diffusion')

    @staticmethod
    def prepare_lr(clip: vs.VideoNode) -> vs.VideoNode:
        """Prepare LR clip in RGB24"""
        clip = depth(clip, 16)

        ups = vdf.scale.to_444(clip, znedi=True)
        if isinstance(ups, list):
            raise TypeError

        return ups.resize.Bicubic(format=vs.RGB24, dither_type='error_diffusion')



class ExportDataset:  # noqa: PLC0115
    def write_image_async(self, dataset: Dataset) -> None:  # noqa: PLC0116
        self._output_images(dataset.lr)
        self._output_images(dataset.hr)

    @staticmethod
    def _output_images(clip_dts: ClipForDataset) -> None:
        if not clip_dts.path.exists():
            clip_dts.path.mkdir(parents=True)

        progress = Progress(TextColumn("{task.description}"), BarColumn(),
                            TextColumn("{task.completed}/{task.total}"),
                            TextColumn("{task.percentage:>3.02f}%"),
                            FPSColumn(), TimeRemainingColumn())

        with progress:
            task = progress.add_task('Extracting frames...', total=clip_dts.clip.num_frames)

            def _cb(n: int, f: vs.VideoFrame) -> None:  # noqa: PLC0103
                progress.update(task, advance=1)

            clip = clip_dts.clip.imwri.Write(
                'PNG', filename=str(clip_dts.path.joinpath('gooya_%06d.png'))
            )

            clip_async_render(clip, callback=_cb)



if __name__ == '__main__':
    # Output ready-to-use dataset
    dts = PrepareDataset().prepare()

    # And write the dataset!
    ExportDataset().write_image_async(dts)

else:
    # Preview
    dts = PrepareDataset().prepare()
    dts.lr.clip.set_output(0)
    dts.hr.clip.set_output(1)
