import numpy as np
import unittest
from jpeg_antecedent.jpeg_toolbox import llm_dct_islow, jpeg_fdct_islow, define_quant_table, quality_scaling_law, \
    quantize_islow_fdct, jpeg_fdct_naive, jpeg_idct_naive, jpeg_idct_islow


class TestJPEGToolbox(unittest.TestCase):

    def test_llm_dct_islow_color(self):
        color_block = np.random.randint(0, 256, size=(8, 8, 1, 3), dtype=np.int32)

        first_color_dct = llm_dct_islow(np.copy(color_block), first_pass=True)
        first_split_dct = np.concatenate([llm_dct_islow(b, first_pass=True)
                                          for b in np.split(color_block, 3, axis=-1)],
                                         axis=-1)
        self.assertTrue(np.allclose(first_color_dct, first_split_dct))

        second_color_dct = llm_dct_islow(np.copy(first_color_dct).transpose(1, 0, 2, 3), first_pass=False)
        second_split_dct = np.concatenate(
            [llm_dct_islow(b.transpose(1, 0, 2, 3), first_pass=False)
             for b in np.split(first_split_dct, 3, axis=-1)],
            axis=-1)
        self.assertTrue(np.allclose(second_color_dct, second_split_dct))

    def test_llm_dct_islow_batch(self):
        n_block = 10
        batch_block = np.random.randint(0, 256, size=(8, 8, n_block, 1), dtype=np.int32)

        first_batch_dct = llm_dct_islow(np.copy(batch_block), first_pass=True)
        first_split_dct = np.concatenate([llm_dct_islow(b, first_pass=True)
                                          for b in np.split(batch_block, n_block, axis=2)],
                                         axis=2)
        self.assertTrue(np.allclose(first_batch_dct, first_split_dct))

        second_batch_dct = llm_dct_islow(np.copy(first_batch_dct).transpose(1, 0, 2, 3), first_pass=False)
        second_split_dct = np.concatenate(
            [llm_dct_islow(b.transpose(1, 0, 2, 3), first_pass=False)
             for b in np.split(first_split_dct, n_block, axis=2)],
            axis=2)
        self.assertTrue(np.allclose(second_batch_dct, second_split_dct))

    def test_jpeg_dct_islow_color(self):
        blocks = np.random.randint(0, 256, size=(1, 3, 8, 8), dtype=np.int32)

        dct_blocks = jpeg_fdct_islow(np.copy(blocks))
        split_dct_blocks = np.concatenate([jpeg_fdct_islow(b) for b in np.split(blocks, 3, axis=1)],
                                          axis=1)
        self.assertTrue(np.allclose(dct_blocks, split_dct_blocks))

    def test_jpeg_dct_islow_batch(self):
        n_blocks = 10
        blocks = np.random.randint(0, 256, size=(n_blocks, 1, 8, 8), dtype=np.int32)

        dct_blocks = jpeg_fdct_islow(np.copy(blocks))
        split_dct_blocks = np.concatenate([jpeg_fdct_islow(b) for b in np.split(blocks, n_blocks, axis=0)],
                                          axis=0)
        self.assertTrue(np.allclose(dct_blocks, split_dct_blocks))

    def test_quantize_islow_color(self):
        luminance, chrominance = define_quant_table(quality_scaling_law(83))
        blocks = np.random.randint(0, 256, size=(1, 3, 8, 8), dtype=np.int32)
        split_blocks = np.split(blocks, 3, axis=1)

        quantized_blocks = quantize_islow_fdct(np.copy(blocks), np.array([luminance, chrominance, chrominance]))
        split_quantized_blocks = np.concatenate([quantize_islow_fdct(split_blocks[0], luminance),
                                                 quantize_islow_fdct(split_blocks[1], chrominance),
                                                 quantize_islow_fdct(split_blocks[2], chrominance)],
                                                axis=1)
        self.assertTrue(np.allclose(quantized_blocks, split_quantized_blocks))

    def test_naive_dct_inversion(self):
        quant_tbl = np.ones((8,8), dtype=np.uint16)
        blocks = np.random.randint(0, 256, size=(1, 1, 8, 8), dtype=np.int32)
        dct = jpeg_fdct_naive(blocks)
        spatial = jpeg_idct_naive(dct, quant_tbl=quant_tbl)
        self.assertTrue(np.allclose(spatial, blocks))