import unittest
import numpy as np
import jpeglib
import os
from PIL import Image
from skimage.util import view_as_blocks

from jpeg_antecedent.pipeline import IslowPipeline


class TestIslowPipeline(unittest.TestCase):
    def setUp(self):
        jpeglib.version.set('6b')
        self.tif_img = np.asarray(Image.open('test_images/spatial_color.tif'))
        self.tif_img = self.tif_img[:256, :256, :3]

        self.start_quality = 24  # quantization tables are too coarse below quality = 24 for baseline JPEG
        self.end_quality = 101
        self.compressed_img = np.zeros((self.end_quality - self.start_quality,
                                        (self.tif_img.shape[0] // 8) * (self.tif_img.shape[1] // 8), 3, 8, 8))
        self.decompressed_img = np.zeros((self.end_quality - self.start_quality,
                                          (self.tif_img.shape[0] // 8) * (self.tif_img.shape[1] // 8), 3, 8, 8))

        islow = jpeglib.DCTMethod(0)

        for quality in range(self.start_quality, self.end_quality):
            jlib_pixel = jpeglib.from_spatial(self.tif_img, jpeglib.Colorspace.JCS_RGB)
            jlib_pixel.samp_factor = [[1, 1], [1, 1], [1, 1]]
            jlib_pixel.write_spatial('test.jpg', qt=quality, dct_method=jpeglib.DCTMethod(0))
            jlib_dct = jpeglib.read_dct('test.jpg')
            idx = quality - self.start_quality

            self.compressed_img[idx] = np.stack([jlib_dct.Y, jlib_dct.Cb, jlib_dct.Cr], axis=2).reshape(-1, 3, 8, 8)
            jlib_decomp_pxl = jpeglib.read_spatial('test.jpg', jpeglib.Colorspace.JCS_RGB, islow)
            self.decompressed_img[idx] = view_as_blocks(jlib_decomp_pxl.spatial, (8, 8, 1)).reshape(-1, 3, 8, 8)

    def tearDown(self):
        if os.path.isfile("test.jpg"):
            os.remove("test.jpg")

    def test_color_compression(self):
        for quality in range(self.start_quality, self.end_quality):
            compressed_img = self.compressed_img[quality - self.start_quality]

            pipeline = IslowPipeline(quality, False, False)
            rgb_blocks = view_as_blocks(np.copy(self.tif_img), (8, 8, 1)).reshape(-1, 3, 8, 8)
            compressed_blocks = pipeline.rgb_pxl_to_ycc_dct(rgb_blocks)

            self.assertTrue(np.allclose(compressed_blocks, compressed_img))

    def test_color_decompression(self):
        for quality in range(self.start_quality, self.end_quality):
            decompressed_img = self.decompressed_img[quality - self.start_quality]

            pipeline = IslowPipeline(quality, False, False)
            dct_blocks = self.compressed_img[quality - self.start_quality]
            decompressed_blocks = pipeline.ycc_dct_to_rgb_pxl(dct_blocks)

            self.assertTrue(np.allclose(decompressed_blocks, decompressed_img))
