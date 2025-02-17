from jpeg_antecedent.jpeg_toolbox import jpeg_fdct_float, jpeg_fdct_ifast, jpeg_fdct_islow, jpeg_fdct_naive, jpeg_idct_naive, \
    quantize_ifast_fdct, quantize_float_fdct, quantize_islow_fdct, quantize_naive_fdct, jpeg_idct_islow, \
    define_quant_table, quality_scaling_law, rgb_to_ycc, ycc_to_rgb, rgb_to_ycc_float, ycc_to_rgb_float
from jpeg_antecedent.utils import round
import numpy as np


class JPEGPipeline:
    def __init__(self, name, quality, target_is_dct, grayscale, return_int=True, return_rounded=True):
        self.upper_bound = None
        self.name = name
        self.quality = quality
        self.target_is_dct = target_is_dct
        self.grayscale = grayscale
        self.quant_tbl = None
        self.return_rounded = return_rounded
        self.define_quant_table()
        self.define_upper_bound()
        if grayscale and target_is_dct and return_int:
            self.forward = self.pxl_to_dct
            self.backward = self.dct_to_pxl
        elif grayscale and target_is_dct and not return_int:
            self.forward = self.pxl_to_dct_float
            self.backward = self.dct_to_pxl_float
        elif grayscale and not target_is_dct and return_int:
            self.forward = self.dct_to_pxl
            self.backward = self.pxl_to_dct
        elif grayscale and not target_is_dct and not return_int:
            self.forward = self.dct_to_pxl_float
            self.backward = self.pxl_to_dct_float
        elif not grayscale and target_is_dct and return_int:
            self.forward = self.rgb_pxl_to_ycc_dct
            self.backward = self.ycc_dct_to_rgb_pxl
        elif not grayscale and target_is_dct and not return_int:
            self.forward = self.rgb_pxl_to_ycc_dct_float
            self.backward = self.ycc_dct_to_rgb_pxl_float
        elif return_int:
            self.forward = self.ycc_dct_to_rgb_pxl
            self.backward = self.rgb_pxl_to_ycc_dct
        else:
            self.forward = self.ycc_dct_to_rgb_pxl_float
            self.backward = self.rgb_pxl_to_ycc_dct_float

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __hash__(self):
        h = 0
        for ch in self.name + str(self.quant_tbl.astype(np.int32)):
            h = (h * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
        return h

    def define_upper_bound(self):
        self.upper_bound = np.zeros((self.quant_tbl.shape[0])).reshape(-1, 1, 1)
        if self.target_is_dct:
            self.upper_bound += np.linalg.norm(self.quant_tbl / 2, axis=(1, 2), ord='fro', keepdims=True)
        else:
            self.upper_bound += 4
        if not self.grayscale:
            self.upper_bound += 4

    @classmethod
    def is_named(cls, name):
        return False

    def define_quant_table(self):
        luminance, chrominance = define_quant_table(quality_scaling_law(self.quality), True)
        if self.grayscale:
            self.quant_tbl = np.expand_dims(luminance, axis=0)
        else:
            self.quant_tbl = np.stack([luminance, chrominance, chrominance])

    def dct(self, blocks):
        pass

    def idct(self, blocks):
        pass

    def rgb_pxl_to_ycc_dct(self, blocks):
        return self.round_dct(self.dct(self.rgb_to_ycc(blocks)))

    def ycc_dct_to_rgb_pxl(self, blocks):
        return self.ycc_to_rgb(self.round_pixel(self.idct(blocks)))

    def pxl_to_dct(self, blocks):
        return self.round_dct(self.dct(blocks))

    def dct_to_pxl(self, blocks):
        return self.round_pixel(self.idct(blocks))

    def rgb_pxl_to_ycc_dct_float(self, blocks):
        return self.dct(self.rgb_to_ycc_float(blocks))

    def ycc_dct_to_rgb_pxl_float(self, blocks):
        return self.ycc_to_rgb_float(self.idct(blocks))

    def pxl_to_dct_float(self, blocks):
        return self.dct(blocks)

    def dct_to_pxl_float(self, blocks):
        return self.idct(blocks)

    def round_dct(self, blocks):
        return round(blocks).astype(np.int32)

    def round_pixel(self, blocks):
        return round(blocks).astype(np.int32)

    def rgb_to_ycc(self, blocks):
        return rgb_to_ycc(blocks)

    def ycc_to_rgb(self, blocks):
        return ycc_to_rgb(blocks)

    def rgb_to_ycc_float(self, blocks):
        return rgb_to_ycc_float(blocks)

    def ycc_to_rgb_float(self, blocks):
        return ycc_to_rgb_float(blocks)


class NaivePipeline(JPEGPipeline):
    def __init__(self, quality, target_is_dct, grayscale, return_int=True, return_rounded=True):
        super().__init__('naive', quality, target_is_dct, grayscale, return_int, return_rounded)

    @classmethod
    def is_named(cls, name):
        return name == 'naive'

    def dct(self, blocks):
        return quantize_naive_fdct(jpeg_fdct_naive(blocks), quant_tbl=self.quant_tbl[:blocks.shape[1]],
                                   return_int=self.return_rounded)

    def idct(self, blocks):
        return jpeg_idct_naive(blocks, self.quant_tbl[:blocks.shape[1]])


class IslowPipeline(JPEGPipeline):
    def __init__(self, quality, target_is_dct, grayscale, return_int=True, return_rounded=True):
        super().__init__('islow', quality, target_is_dct, grayscale, return_int, return_rounded)
        self.return_int = return_int

    @classmethod
    def is_named(cls, name):
        return name == 'islow'

    def dct(self, blocks):
        return quantize_islow_fdct(jpeg_fdct_islow(blocks), quant_tbl=self.quant_tbl[:blocks.shape[1]],
                                   return_int=False)

    def idct(self, blocks):
        return jpeg_idct_islow(blocks, self.quant_tbl[:blocks.shape[1]])

    def pxl_to_dct(self, blocks):
        return quantize_islow_fdct(jpeg_fdct_islow(blocks), quant_tbl=self.quant_tbl[:blocks.shape[1]],
                                   return_int=self.return_rounded)

    def rgb_pxl_to_ycc_dct(self, blocks):
        return quantize_islow_fdct(jpeg_fdct_islow(rgb_to_ycc(blocks)), quant_tbl=self.quant_tbl,
                                   return_int=self.return_rounded)

    def dct_to_pxl(self, blocks):
        return jpeg_idct_islow(blocks, self.quant_tbl[:blocks.shape[1]])

    def ycc_dct_to_rgb_pxl(self, blocks):
        return ycc_to_rgb(jpeg_idct_islow(blocks, self.quant_tbl))


class IfastPipeline(JPEGPipeline):
    def __init__(self, quality, target_is_dct, grayscale, return_int=True, return_rounded=True):
        super().__init__('ifast', quality, target_is_dct, grayscale, return_int, return_rounded)

    @classmethod
    def is_named(cls, name):
        return name == 'ifast'

    def dct(self, blocks):
        return quantize_ifast_fdct(jpeg_fdct_ifast(blocks - 128), quant_tbl=self.quant_tbl[:blocks.shape[1]],
                                   return_int=self.return_rounded)

    def idct(self, blocks):
        return jpeg_idct_naive(blocks, self.quant_tbl[:blocks.shape[1]]) + 128  # jpeg_idct_fast not implemented yet


class FloatPipeline(JPEGPipeline):
    def __init__(self, quality, target_is_dct, grayscale, return_int=True, return_rounded=True):
        super().__init__('float', quality, target_is_dct, grayscale, return_int, return_rounded)

    @classmethod
    def is_named(cls, name):
        return name == 'float'

    def dct(self, blocks):
        return quantize_float_fdct(jpeg_fdct_float(blocks - 128), quant_tbl=self.quant_tbl[:blocks.shape[1]],
                                   return_int=self.return_rounded)

    def idct(self, blocks):
        return jpeg_idct_naive(blocks, self.quant_tbl[:blocks.shape[1]]) + 128  # jpeg_idct_float not implemented yet


class ComposedPipeline:
    def __init__(self, pipelines, backward_naive=True):
        self.pipelines = pipelines
        self.upper_bound = np.sum([pipe.upper_bound for pipe in pipelines], axis=0) + 1
        if backward_naive:
            self.backward_pipes = [
                NaivePipeline(pipe.quality, pipe.target_is_dct, pipe.grayscale, return_int=False, return_rounded=False)
                for pipe in pipelines]
        else:
            self.backward_pipes = self.pipelines
        self.n = len(pipelines)

    def __hash__(self):
        return tuple(self.pipelines).__hash__()

    def __repr__(self):
        return tuple(self.pipelines).__repr__()

    def __str__(self):
        return tuple(self.pipelines).__str__()

    def forward(self, blocks):
        for pipe in self.pipelines:
            blocks = pipe.forward(blocks)
        return blocks

    def backward(self, blocks):
        for pipe in self.backward_pipes[::-1]:
            blocks = pipe.backward(blocks)
        return blocks


def create_pipeline(name, quality, grayscale, target_is_dct, backward_naive=True, return_int=True, return_rounded=True):
    if not isinstance(name, list):
        name = [name]
    if not isinstance(quality, list):
        quality = [quality]

    if len(name) != len(quality):
        raise ValueError(f"In the config.yaml, pipeline name and quality must have the same length")

    pipelines = []
    for i in range(len(name)):
        found = False
        if (len(name) - i - 1) % 2 == 0:  # ensure that the last pipeline will be used for compression if return_dct
            compression = target_is_dct
        else:
            compression = not target_is_dct
        for subclass in JPEGPipeline.__subclasses__():
            if subclass.is_named(name[i].lower()):
                pipelines.append(
                    subclass(quality=quality[i], target_is_dct=compression, grayscale=grayscale, return_int=return_int,
                             return_rounded=return_rounded))
                found = True
                break
        if found:
            continue
        else:
            raise ValueError(f"Could not find the pipeline {name[i]} in the registered pipeline.")
    return ComposedPipeline(pipelines, backward_naive)
