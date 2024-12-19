import unittest
import numpy as np
from PIL import Image as pilimage

from jpeg_antecedent.image import Image
from jpeg_antecedent.pipeline import create_pipeline

class TestOpen(unittest.TestCase):
    def setUp(self):
        self.spatial_color = "test_images/spatial_color.tif"
        self.jpeg_color = "test_images/jpeg_color.jpg"
        self.spatial_grayscale = "test_images/spatial_grayscale.tif"
        self.jpeg_grayscale = "test_images/jpeg_grayscale.jpg"

        with pilimage.open(self.spatial_color) as img:
            self.pil_spatial_color = img
        with pilimage.open(self.jpeg_color) as img:
            self.pil_jpeg_color = img
        with pilimage.open(self.spatial_grayscale) as img:
            self.pil_spatial_grayscale = img
        with pilimage.open(self.jpeg_grayscale) as img:
            self.pil_jpeg_grayscale = img

    def test_jpeg_color(self):
        img = Image(self.jpeg_color, self.jpeg_color)
        standard_shape = (self.pil_jpeg_color.height // 8, self.pil_jpeg_color.width // 8, 3, 8, 8)
        self.assertFalse(img.is_grayscale)
        self.assertTrue(img.channel == 3)
        self.assertTrue(img.block_view.shape == standard_shape)

    def test_jpeg_grayscale(self):
        img = Image(self.jpeg_grayscale, self.jpeg_grayscale)
        standard_shape = (self.pil_jpeg_color.height // 8, self.pil_jpeg_color.width // 8, 1, 8, 8)
        self.assertTrue(img.is_grayscale)
        self.assertTrue(img.channel == 1)
        self.assertTrue(img.block_view.shape == standard_shape)

    def test_spatial_color(self):
        img = Image(self.spatial_color, self.spatial_color)
        standard_shape = (self.pil_spatial_color.height // 8, self.pil_spatial_color.width // 8, 3, 8, 8)
        self.assertFalse(img.is_grayscale)
        self.assertTrue(img.channel == 3)
        self.assertTrue(img.block_view.shape == standard_shape)

    def test_spatial_grayscale(self):
        img = Image(self.spatial_grayscale, self.spatial_grayscale)
        standard_shape = (self.pil_spatial_grayscale.height // 8, self.pil_spatial_grayscale.width // 8, 1, 8, 8)
        self.assertTrue(img.is_grayscale)
        self.assertTrue(img.channel == 1)
        self.assertTrue(img.block_view.shape == standard_shape)


class TestFilterJPEGColor(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/jpeg_color.jpg"
        self.img = Image(self.filename, self.filename)

        self.img.set_filter_parameter(True)
        self.simple_pipeline = create_pipeline('islow', 60, False, True)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], False, True)
        self.triple_pipeline = create_pipeline(['islow', 'islow', 'islow'], [75, 75, 60], False, True)

    def test_simple_pipeline(self):
        backward_blocks = self.simple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                     np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.simple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_double_pipeline(self):
        self.img.set_pipeline(self.simple_pipeline)
        mask_simple = self.img.filter_blocks(purge=True)
        self.img.set_pipeline(self.double_pipeline)
        mask_double = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask_simple, mask_double))

    def test_triple_pipeline(self):
        backward_blocks = self.simple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                     np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        backward_blocks = self.triple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = true_mask | (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                                 np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.triple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))


class TestFilterJPEGGrayscale(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/jpeg_grayscale.jpg"
        self.img = Image(self.filename, self.filename)

        self.img.set_filter_parameter(True)
        self.simple_pipeline = create_pipeline('islow', 60, True, True)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], True, True)
        self.triple_pipeline = create_pipeline(['islow', 'islow', 'islow'], [75, 75, 60], True, True)

    def test_simple_pipeline(self):
        backward_blocks = self.simple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                     np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.simple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_double_pipeline(self):
        self.img.set_pipeline(self.simple_pipeline)
        mask_simple = self.img.filter_blocks(purge=True)
        self.img.set_pipeline(self.double_pipeline)
        mask_double = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask_simple, mask_double))

    def test_triple_pipeline(self):
        backward_blocks = self.simple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                     np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        backward_blocks = self.triple_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = true_mask | (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                                 np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.triple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))


class TestFilterSpatialColor(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/spatial_color.tif"
        self.img = Image(self.filename, self.filename)

        self.img.set_filter_parameter(True)
        self.simple_pipeline = create_pipeline('islow', 60, False, False)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], False, False)
        self.triple_pipeline = create_pipeline(['islow', 'islow', 'islow'], [75, 75, 60], False, False)

    def test_simple_pipeline(self):
        blocks = self.img.block_view.reshape(-1, self.img.channel, 8, 8)
        true_mask = (np.any(blocks <= 0, axis=(1, 2, 3)) |
                     np.any(blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.simple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_double_pipeline(self):
        blocks = self.img.block_view.reshape(-1, self.img.channel, 8, 8)
        true_mask = (np.any(blocks <= 0, axis=(1, 2, 3)) |
                     np.any(blocks >= 255, axis=(1, 2, 3)))
        backward_blocks = self.double_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = true_mask | (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                                 np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        print(true_mask.shape)
        print(true_mask)
        self.img.set_pipeline(self.simple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        print(mask.shape)
        print(mask)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_triple_pipeline(self):
        self.img.set_pipeline(self.double_pipeline)
        mask_simple = self.img.filter_blocks(purge=True)
        self.img.set_pipeline(self.triple_pipeline)
        mask_double = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask_simple, mask_double))

class TestFilterSpatialGrayscale(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/spatial_grayscale.tif"
        self.img = Image(self.filename, self.filename)

        self.img.set_filter_parameter(True)
        self.simple_pipeline = create_pipeline('islow', 60, True, False)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], True, False)
        self.triple_pipeline = create_pipeline(['islow', 'islow', 'islow'], [75, 75, 60], True, False)

    def test_simple_pipeline(self):
        blocks = self.img.block_view.reshape(-1, self.img.channel, 8, 8)
        true_mask = (np.any(blocks <= 0, axis=(1, 2, 3)) |
                     np.any(blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.simple_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_double_pipeline(self):
        blocks = self.img.block_view.reshape(-1, self.img.channel, 8, 8)
        true_mask = (np.any(blocks <= 0, axis=(1, 2, 3)) |
                     np.any(blocks >= 255, axis=(1, 2, 3)))
        backward_blocks = self.double_pipeline.backward(self.img.block_view.reshape(-1, self.img.channel, 8, 8))
        true_mask = true_mask | (np.any(backward_blocks <= 0, axis=(1, 2, 3)) |
                                 np.any(backward_blocks >= 255, axis=(1, 2, 3)))
        true_mask = true_mask.reshape(self.img.block_view.shape[:2])
        self.img.set_pipeline(self.double_pipeline)
        mask = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask, true_mask))

    def test_triple_pipeline(self):
        self.img.set_pipeline(self.double_pipeline)
        mask_simple = self.img.filter_blocks(purge=True)
        self.img.set_pipeline(self.triple_pipeline)
        mask_double = self.img.filter_blocks(purge=True)
        self.assertTrue(np.allclose(mask_simple, mask_double))

class TestSelectionVarianceJPEGColor(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/jpeg_color.jpg"
        self.img = Image(self.filename, self.filename)
        self.simple_pipeline = create_pipeline('islow', 60, False, True)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], False, True)
        self.img.set_selection_parameter("variance", 20)

    def test_selection(self):
        self.img.set_pipeline(self.simple_pipeline)
        self.img.filter_blocks(purge=True)
        self.img.select_blocks(purge=True)

class TestSelectionVarianceJPEGGrayscale(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/jpeg_grayscale.jpg"
        self.img = Image(self.filename, self.filename)
        self.simple_pipeline = create_pipeline('islow', 60, True, True)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], True, True)
        self.img.set_selection_parameter("variance", 20)

    def test_selection(self):
        self.img.set_pipeline(self.simple_pipeline)
        self.img.filter_blocks(purge=True)
        self.img.select_blocks(purge=True)

class TestSelectionVarianceSpatialColor(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/spatial_color.tif"
        self.img = Image(self.filename, self.filename)
        self.simple_pipeline = create_pipeline('islow', 60, False, False)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], False, False)
        self.img.set_selection_parameter("variance", 20)

    def test_selection(self):
        self.img.set_pipeline(self.simple_pipeline)
        self.img.filter_blocks(purge=True)
        self.img.select_blocks(purge=True)

class TestSelectionVarianceSpatialGrayscale(unittest.TestCase):
    def setUp(self):
        self.filename = "test_images/spatial_grayscale.tif"
        self.img = Image(self.filename, self.filename)
        self.simple_pipeline = create_pipeline('islow', 60, True, False)
        self.double_pipeline = create_pipeline(['islow', 'islow'], [75, 60], True, False)
        self.img.set_selection_parameter("variance", 20)

    def test_selection(self):
        self.img.set_pipeline(self.simple_pipeline)
        self.img.filter_blocks(purge=True)
        self.img.select_blocks(purge=True)