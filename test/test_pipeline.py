import unittest
import numpy as np
import jpeglib
import os

from PIL import Image
from skimage.util import view_as_blocks
from itertools import product

from jpeg_antecedent.pipeline import create_pipeline


def split_into_blocks(img):
    return view_as_blocks(img, (8, 8, 1))


def flatten_blocks(img, colorspace):
    nb_channels = 3 if colorspace.name == jpeglib.Colorspace.JCS_RGB.name else 1
    return img.reshape(-1, nb_channels, 8, 8)


def jpeglib_compress(img, colorspace, quality, dct_method, samp_factor, version):
    jpeglib.version.set(version)
    temp_filename = "tmp.jpg"  # get_temp_filename()
    jpeglib_pixel = jpeglib.from_spatial(img, colorspace)
    jpeglib_pixel.samp_factor = samp_factor
    jpeglib_pixel.write_spatial(temp_filename, qt=quality, dct_method=jpeglib.DCTMethod(dct_method))

    jpeglib_dct = jpeglib.read_dct(temp_filename)

    jpeglib_dct = [channel for channel in [jpeglib_dct.Y, jpeglib_dct.Cb, jpeglib_dct.Cr] if channel is not None]
    jpeglib_dct = np.stack(jpeglib_dct, axis=2)

    return flatten_blocks(jpeglib_dct, colorspace)


def jpeglib_decompress(temp_filename, colorspace, dct_method, version):
    jpeglib.version.set(version)

    jpeglib_pixel = jpeglib.read_spatial(temp_filename, colorspace, jpeglib.DCTMethod(dct_method))

    return flatten_blocks(split_into_blocks(jpeglib_pixel.spatial), colorspace)


def create_pipeline_from_params(quality, dct_method, colorspace, compression):
    if dct_method == 0:
        name = 'islow'
    elif dct_method == 1:
        name = 'ifast'
    else:
        name = 'float'
    if colorspace.name == jpeglib.Colorspace.JCS_RGB.name:
        grayscale = False
    else:
        grayscale = True
    if compression:
        return create_pipeline(name, quality, grayscale=grayscale, target_is_dct=True)
    return create_pipeline(name, quality, grayscale=grayscale, target_is_dct=False)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        n = 256
        with Image.open('test_images/spatial_color.tif') as img:
            self.img_color = np.asarray(img)[:n, :n]
            self.img_grayscale = np.expand_dims(np.asarray(img.convert('L'))[:n, :n], axis=-1)
        self.quality = np.arange(24, 101).tolist()
        self.dct_method = [0, 2]
        self.colorspace = [jpeglib.Colorspace.JCS_RGB, jpeglib.Colorspace.JCS_GRAYSCALE]
        self.version = ['6b']
        self.samp_factor = [[[1, 1], [1, 1], [1, 1]]]
        self.temp_filename = "tmp.jpg"

    def tearDown(self):
        if os.path.exists(self.temp_filename):
            os.remove(self.temp_filename)

    def get_img(self, colorspace):
        if colorspace.name == jpeglib.Colorspace.JCS_RGB.name:
            return self.img_color
        return self.img_grayscale

    def compression_test(self, quality, dct_method, colorspace, samp_factor, version):
        compression_pipeline = create_pipeline_from_params(quality, dct_method, colorspace, compression=True)
        img = self.get_img(colorspace)

        jpeglib_dct = jpeglib_compress(img, colorspace, quality, dct_method, samp_factor, version)
        dct = compression_pipeline.forward(flatten_blocks(split_into_blocks(img), colorspace))
        self.assertTrue(np.allclose(dct, jpeglib_dct), "Compression failed")
        return dct

    def decompression_test(self, quality, dct_method, colorspace, version, dct):
        decompression_pipeline = create_pipeline_from_params(quality, dct_method, colorspace, compression=False)
        jpeglib_pixel = jpeglib_decompress(self.temp_filename, colorspace, dct_method, version)
        pixel = decompression_pipeline.forward(dct)
        self.assertTrue(np.allclose(pixel, jpeglib_pixel), "Decompression failed")

    def test_pipeline(self):
        for params in product(self.quality, self.dct_method, self.colorspace, self.samp_factor, self.version):
            quality, dct_method, colorspace, samp_factor, version = params
            with self.subTest(compression=True, quality=quality, dct_method=dct_method, colorspace=colorspace,
                              samp_factor=samp_factor, version=version):
                dct = self.compression_test(quality, dct_method, colorspace, samp_factor, version)
            with self.subTest(compression=False, quality=quality, dct_method=dct_method, colorspace=colorspace,
                              version=version):
                self.decompression_test(quality, dct_method, colorspace, version, dct)
