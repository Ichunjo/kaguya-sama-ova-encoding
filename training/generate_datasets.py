"""Generate a dataset for use in BasicSR"""


import os
import random
import subprocess
from pathlib import Path
from typing import BinaryIO, NamedTuple, Set, Tuple, cast

import vapoursynth as vs
import vardefunc as vdf
from lvsfunc.misc import source
from lvsfunc.progress import (BarColumn, FPSColumn, Progress, TextColumn,
                              TimeRemainingColumn)
from lvsfunc.render import clip_async_render
from vsutil import depth

core = vs.core


# First import your clips
CLIP_DVD = source('../GOOYA_S2_DVDISO/KAGUYA_S2_VOL1/VIDEO_TS/VTS_01_1.dgi', fieldop=1) \
    + source('../GOOYA_S2_DVDISO/KAGUYA_S2_VOL2/VIDEO_TS/VTS_01_1.dgi', fieldop=1) \
    + source('../GOOYA_S2_DVDISO/KAGUYA_S2_VOL3/VIDEO_TS/VTS_01_1.dgi', fieldop=1)
CLIP_BD_ = source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00001.dgi') \
    + source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~1/BDMV/BDMV/STREAM/00002.dgi') \
    + source('../GOOYA_S2_BDMV/かぐや様は告らせたい？～天才たちの恋愛頭脳戦～2/BDMV/BDMV/STREAM/00001.dgi') \
    + source('../GOOYA_S2_BDMV/かぐや様は告らせたい？～天才たちの恋愛頭脳戦～2/BDMV/BDMV/STREAM/00002.dgi') \
    + source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~3/BDMV/BDMV/STREAM/00001.dgi') \
    + source('../GOOYA_S2_BDMV/かぐや様は告らせたい~天才たちの恋愛頭脳戦~3/BDMV/BDMV/STREAM/00002.dgi')



# Since both DVD and BD doesn't have the matrix props, we need to add them.
# If not, colours won't match.

# _FieldBased is also added since Kaguya is a 2:3 pulldown
# and we removed this flag with dgdecodenv.DGSource
PROPS_DVD: Set[Tuple[str, int]] = {
    ('_ChromaLocation', 0),
    ('_Matrix', 5),
    ('_Transfer', 5),
    ('_Primaries', 5),
    ('_FieldBased', 0)
}

PROPS_BD: Set[Tuple[str, int]] = {
    ('_Matrix', 1),
    ('_Transfer', 1),
    ('_Primaries', 1)
}


class ClipForDatasets(NamedTuple):  # noqa: PLC0115
    clip: vs.VideoNode
    res_type: str


class Datasets(NamedTuple):  # noqa: PLC0115
    hr: ClipForDatasets
    lr: ClipForDatasets


PATH_DATASET = Path('dataset')
PATH_DATASET_TRAIN = PATH_DATASET.joinpath('train')
PATH_DATASET_VAL = PATH_DATASET.joinpath('val')


class PrepareDataset:  # noqa: PLC0115
    def prepare(self) -> Datasets:  # noqa: PLC0116
        # Make your adjustments to match the framerates and frames
        dvd = CLIP_DVD
        bd_ = CLIP_BD_

        print('Cut...\n')
        dvd = dvd[:70513] + dvd[86572:]
        dvd = dvd[:139592] + dvd[151855:]
        dvd = dvd[:171962] + dvd[174144:204354]

        bd_ = bd_[:70513] + bd_[70540:]
        bd_ = bd_[:139592] + bd_[141775:]
        bd_ = bd_[:171962] + bd_[174144:204354]


        # Force props
        for prop, val in PROPS_DVD:
            dvd = dvd.std.SetFrameProp(prop, intval=val)

        for prop, val in PROPS_BD:
            bd_ = bd_.std.SetFrameProp(prop, intval=val)

        # Prepare the HR and LR clip for dataset
        print('Prepare HR RGB24...\n')
        hr_ = self.prepare_hr(bd_)
        print('Prepare LR RGB24...\n')
        lr_ = self.prepare_lr(dvd)


        if hr_.format != lr_.format:
            raise vs.Error('Format of LR and HR do not match!')



        # We select 5 % of frames to make a reasonable size dataset
        if (length := dvd.num_frames) == dvd.num_frames:
            frames = sorted(random.sample(population=range(length), k=round(length*0.05)))
        else:
            raise IndexError("LR and HR don't have the same length!")

        # Select frames
        print('Select frames...\n')
        hr_ = core.std.Splice([hr_[f] for f in frames])
        lr_ = core.std.Splice([lr_[f] for f in frames])


        return Datasets(hr=ClipForDatasets(hr_, 'HR'),
                        lr=ClipForDatasets(lr_, 'LR'))


    @staticmethod
    def prepare_hr(clip: vs.VideoNode) -> vs.VideoNode:
        """Prepare HR clip in RGB24"""
        clip = depth(clip, 16).std.AssumeFPS(fpsnum=24, fpsden=1)

        ups = vdf.scale.to_444(clip, znedi=True)
        if isinstance(ups, list):
            raise TypeError

        # The DVD is oddly cropped. We need to adjust the HR clip for a better training.
        return ups.std.Crop(top=6, bottom=7).resize.Bicubic(
            CLIP_DVD.width*2, CLIP_DVD.height*2, vs.RGB24,
            src_left=-0.5, src_top=0.5, dither_type='error_diffusion'
            # And swap the planes for gbr output
        ).std.ShufflePlanes([1, 2, 0], vs.RGB)

    @staticmethod
    def prepare_lr(clip: vs.VideoNode) -> vs.VideoNode:
        """Prepare LR clip in RGB24"""
        clip = depth(clip, 16).std.AssumeFPS(fpsnum=24, fpsden=1)

        ups = vdf.scale.to_444(clip, znedi=True)
        if isinstance(ups, list):
            raise TypeError

        return ups.resize.Bicubic(
            format=vs.RGB24, dither_type='error_diffusion'
            # And swap the planes for gbr output
        ).std.ShufflePlanes([1, 2, 0], vs.RGB)



class ExportDataset:  # noqa: PLC0115
    def write_image_async(self, dataset: Datasets) -> None:  # noqa: PLC0116
        # This method is slower :(
        print('Extract LR...\n')
        self._output_images(dataset.lr)
        print('Extract HR...\n')
        self._output_images(dataset.hr)

    @staticmethod
    def _output_images(clip_dts: ClipForDatasets) -> None:
        if not (path := PATH_DATASET_TRAIN.joinpath(clip_dts.res_type)).exists():
            path.mkdir(parents=True)

        # Pretty progress bar
        progress = Progress(TextColumn("{task.description}"), BarColumn(),
                            TextColumn("{task.completed}/{task.total}"),
                            TextColumn("{task.percentage:>3.02f}%"),
                            FPSColumn(), TimeRemainingColumn())

        with progress:
            task = progress.add_task('Extracting frames...', total=clip_dts.clip.num_frames)

            def _cb(n: int, f: vs.VideoFrame) -> None:  # noqa: PLC0103
                progress.update(task, advance=1)

            clip = clip_dts.clip.imwri.Write(
                'PNG', filename=str(path.joinpath('%06d.png'))
            )

            clip_async_render(clip, callback=_cb)





    def write_video(self, dataset: Datasets) -> None:  # noqa: PLC0116
        print('Encode and extract LR...\n')
        self._encode_and_extract(dataset.lr)
        print('Encode and extract HR...\n')
        self._encode_and_extract(dataset.hr)


    @staticmethod
    def _encode_and_extract(clip_dts: ClipForDatasets) -> None:
        if not (path := PATH_DATASET_TRAIN.joinpath(clip_dts.res_type)).exists():
            path.mkdir(parents=True)

        params = [
            'ffmpeg', '-hide_banner', '-f', 'rawvideo',
            '-video_size', f'{clip_dts.clip.width}x{clip_dts.clip.height}',
            '-pixel_format', 'gbrp', '-framerate', str(clip_dts.clip.fps),
            '-i', 'pipe:',
            path.joinpath('%06d.png')
        ]

        print('Encoding...\n')
        with subprocess.Popen(params, stdin=subprocess.PIPE) as process:
            clip_dts.clip.output(cast(BinaryIO, process.stdin))


    @staticmethod
    def select_val_images(dataset: Datasets, number: int) -> None:  # noqa: PLC0116
        if not (path_val_hr := PATH_DATASET_VAL.joinpath(dataset.hr.res_type)).exists():
            path_val_hr.mkdir(parents=True)
        if not (path_val_lr := PATH_DATASET_VAL.joinpath(dataset.lr.res_type)).exists():
            path_val_lr.mkdir(parents=True)

        if not (path_train_hr := PATH_DATASET_TRAIN.joinpath(dataset.hr.res_type)).exists():
            raise FileNotFoundError(f'{path_train_hr} not found')
        if not (path_train_lr := PATH_DATASET_TRAIN.joinpath(dataset.lr.res_type)).exists():
            raise FileNotFoundError(f'{path_train_lr} not found')

        images_path = sorted(path_train_hr.glob('*.png'))
        image_idx = random.sample(population=range(len(images_path)), k=number)

        for i in image_idx:
            name = images_path[i].name
            os.system(f'copy "{path_train_hr.joinpath(name)}" "{path_val_hr.joinpath(name)}"')
            os.system(f'copy "{path_train_lr.joinpath(name)}" "{path_val_lr.joinpath(name)}"')



if __name__ == '__main__':
    # Output ready-to-use dataset
    dts = PrepareDataset().prepare()

    # And write the dataset!
    # ExportDataset().write_image_async(dts)
    ExportDataset().write_video(dts)
    ExportDataset.select_val_images(dts, 20)

else:
    # Preview
    dts = PrepareDataset().prepare()

    CLIP_DVD.set_output(0)
    CLIP_BD_.set_output(1)
    # dts.lr.clip.resize.Point(720*8, 480*8).text.FrameProps().set_output(0)
    # dts.hr.clip.resize.Bicubic(720, 480).resize.Point(720*8, 480*8).text.FrameProps().set_output(1)
    dts.lr.clip.text.FrameProps().set_output(2)
    dts.hr.clip.resize.Bicubic(720, 480).text.FrameProps().set_output(3)
