import os

import jpeglib
from PIL import Image as Img
from skimage.util import view_as_blocks
import numpy as np

from jpeg_antecedent.block import Block
from jpeg_antecedent.pipeline import NaivePipeline, create_pipeline
from jpeg_antecedent import utils


class Image:
    """
    A class to read images (JPEG or pixels) and stored them into a collection of blocks.

    This class can be useful for:
     - converting RGB image into YCbCr or grayscale (not implemented)
     - detecting the grid of a decompressed image (not implemented)
     - detecting the quantization table of a decompressed image (not implemented)
     - selecting most interesting blocks among the image (not implemented)
    """

    def __init__(self, filename, path):
        self.filename = filename
        self.pipeline = None
        self.avoid_uniform = True
        self.avoid_clipped = True
        self.selection_percentage = 100
        self.method_name = 'random'
        self.pmap = None
        self.rng = None
        self.img = None
        self.block_collection = {}
        self.is_jpeg = False
        self.selected_blocks = []
        self.is_filtered = False
        self.block_view = None
        self.channel = 0

        ext = os.path.splitext(path)[-1].lower()

        if ext in ['.jpg', '.jpeg']:
            self.is_jpeg = True
            self.img = jpeglib.read_dct(path)
            self.is_grayscale = not self.img.has_chrominance
            if self.is_grayscale:
                self.data = np.expand_dims(self.img.Y, 2)
            else:
                self.data = np.stack([self.img.Y, self.img.Cb, self.img.Cr], axis=2)  # shape (None, None, 3, 8, 8)
            self.channel = self.data.shape[2]
            self.block_view = np.copy(self.data)
        else:
            self.img = Img.open(path)
            self.data = np.asarray(self.img)
            self.is_grayscale = self.img.mode == 'L'
            cropped_img = self.data[:8 * (self.data.shape[0] // 8), :8 * (self.data.shape[1] // 8)]
            if self.is_grayscale:
                self.channel = 1
                m, n = cropped_img.shape
            else:
                m, n, self.channel = cropped_img.shape[:3]
            self.block_view = view_as_blocks(cropped_img.reshape(m, n, self.channel), (8, 8, 1))
            self.block_view = self.block_view.reshape(m // 8, n // 8, self.channel, 8, 8)

        for i in range(self.block_view.shape[0]):
            for j in range(self.block_view.shape[1]):
                self.block_collection[(i, j)] = Block(self.block_view[i, j].reshape(self.channel, 8, 8))

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def set_filter_parameter(self, avoid_clipped):
        self.avoid_clipped = avoid_clipped

    def set_selection_parameter(self, method_name, percentage, pmap=None, rng=None):
        if method_name == 'variance' and not self.is_jpeg:
            print("Warning: Selection method 'variance' is only implemented for JPEG images. Using 'random' instead.")
            method_name = 'random'
        self.method_name = method_name
        self.selection_percentage = percentage
        self.pmap = pmap
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()

    def search_antecedent(self, max_iter, shared_dict=None, task_id=None, verbose=False, rng=None):
        self.filter_blocks()

        if self.pipeline.n == 1 and self.is_jpeg and not self.is_grayscale:
            return self.search_antecedent_independent_channel(max_iter, shared_dict, task_id, verbose)

        blocks = self.select_blocks()
        self.trivial_search(blocks)
        n = len(blocks)
        for i, block in enumerate(blocks):
            self.update_shared_dict(shared_dict, task_id, i, n)
            if self.pipeline not in block.status:
                block.search_antecedent(self.pipeline, max_iter, shared_dict, task_id, verbose, rng)

    def update_shared_dict(self, shared_dict, task_id, i, n, completed=0, total=0):
        if shared_dict is not None and task_id is not None:
            shared_dict[task_id] = {"filename": self.filename,
                                    "current_block": i + 1,
                                    "total_block": n,
                                    "completed": completed,
                                    "total": total,
                                    "started": True}

    def search_antecedent_independent_channel(self, max_iter, shared_dict=None, task_id=None, verbose=False, rng=None):
        names = [pipe.name for pipe in self.pipeline]
        qualities = [pipe.quality for pipe in self.pipeline]
        grayscale_pipeline = create_pipeline(names, qualities, grayscale=True, target_is_dct=True)
        blocks = self.select_blocks()
        single_channel_blocks = [Block(np.array([channel])) for block in blocks for channel in block]
        self.trivial_search(single_channel_blocks, grayscale_pipeline)

        antecedents = []
        iterations = []
        all_status = []
        n = len(single_channel_blocks)
        for i, block in enumerate(single_channel_blocks):
            self.update_shared_dict(shared_dict, task_id, i, n)
            if block.status[grayscale_pipeline] == 1:  # Solved with the trivial search
                antecedents.append(block.antecedents[grayscale_pipeline])
                all_status.append(1)
                iterations.append(0)
                continue

            antecedent = block.search_antecedent(grayscale_pipeline, max_iter, shared_dict, task_id, verbose, rng)
            iteration = block.iterations[grayscale_pipeline]
            if antecedent is None:
                antecedent = -np.ones((1, 8, 8))
                status = 0
            elif antecedent == False:
                status = -1
            else:
                status = 1
            antecedents.append(antecedent)
            iterations.append(iteration)
            all_status.append(status)

        iterations = np.sum(np.array(iterations).reshape(len(blocks), 3), axis=-1)
        antecedents = np.array(antecedents).reshape(len(blocks), 3, 8, 8)
        status = np.min(np.array(all_status).reshape(len(blocks), 3), axis=-1)

        for block in blocks:
            block.antecedents[self.pipeline] = np.stack(antecedents, axis=0)
            block.iterations[self.pipeline] = np.sum(iterations)
            block.status[self.pipeline] = status

    def trivial_search(self, blocks, shared_dict=None, task_id=None, pipeline=None):
        if pipeline is None:
            pipeline = self.pipeline
        block_values = np.stack([block.value.reshape(-1, 8, 8) for block in blocks])
        starts = utils.round(pipeline.backward(block_values))
        transformed_starts = pipeline.forward(starts)
        completed = 0
        for i in range(len(starts)):
            if np.allclose(block_values[i], transformed_starts[i]):
                completed += 1
                self.update_shared_dict(shared_dict, task_id, i, len(starts), completed=completed, total=len(starts))
                blocks[i].antecedents[pipeline] = starts[i]
                blocks[i].iterations[pipeline] = 0
                blocks[i].status[pipeline] = 1

    def search_antecedent_ilp(self, parameters, shared_dict, task_id, verbose):
        if (not self.is_jpeg
                or not isinstance(self.pipeline, NaivePipeline)
                or not np.allclose(self.pipeline.quant_tbl, np.ones(self.pipeline.quant_tbl.shape))):
            raise ValueError('Gurobi search is only effective for JPEG images at QF100 with the naive pipeline.')

        self.filter_blocks()
        blocks = self.select_blocks()

        gurobi_parameters = {'IterationLimit': parameters['max_iteration_per_block'],
                             'MIPFocus': parameters['mip_focus'],
                             'Threads': parameters['threads'],
                             'NodefileStart': parameters['node_file_start'],
                             'Cutoff': parameters['cutoff']}
        for i, block in enumerate(blocks):
            self.update_shared_dict(shared_dict, task_id, i, len(blocks), 0, 0)
            block.search_antecedent_ilp(gurobi_parameters, shared_dict, task_id, verbose)

    def classify(self, likelihood_file):
        # TODO: add the LRT test

        pass

    def filter_blocks(self, purge=False):
        if self.is_filtered and not purge:
            return

        is_pixel = not self.is_jpeg
        backward_blocks = np.copy(self.block_view).reshape(-1, self.channel, 8, 8)  # List of blocks (None, c, 8, 8)
        if is_pixel:
            clip_mask = np.any(backward_blocks <= 0, axis=(1, 2, 3)) | np.any(backward_blocks >= 255, axis=(1, 2, 3))
        else:
            clip_mask = np.zeros(backward_blocks.shape[0], dtype=bool)
        for pipe in self.pipeline.backward_pipes[::-1]:
            backward_blocks = pipe.backward(backward_blocks)
            is_pixel = not is_pixel
            if is_pixel:
                clip_mask = clip_mask | np.any(backward_blocks <= 0, axis=(1, 2, 3)) | np.any(backward_blocks >= 255,
                                                                                              axis=(1, 2, 3))
        clip_mask = clip_mask.reshape(self.block_view.shape[:2])
        for pos, block in self.block_collection.items():
            block.is_clipped[self.pipeline] = clip_mask[pos]
        self.is_filtered = True
        return clip_mask

    def select_blocks(self, purge=False):
        if np.any(self.selected_blocks) and not purge:
            return self.selected_blocks
        blocks = [b for _, b in self.block_collection.items() if
                  not (self.avoid_clipped and b.is_clipped[self.pipeline])]

        if self.selection_percentage == 100:
            self.selected_blocks = blocks
            return self.selected_blocks

        n = min(int(len(self.block_collection) * self.selection_percentage / 100), len(blocks))

        if self.method_name == 'random':
            self.selected_blocks = self.rng.choice(blocks, size=n, replace=False)
            return self.selected_blocks

        elif self.method_name == 'variance':
            tmp_blocks = np.stack([b.value for b in blocks]).reshape(-1, self.channel, 8, 8)
            decompressed_blocks = self.pipeline.backward_pipes[-1].backward(np.copy(tmp_blocks))
            rounding_error = np.abs(decompressed_blocks - utils.round(decompressed_blocks))
            variance = np.var(rounding_error, axis=(1, 2, 3))
            var_idx = tmp_blocks.shape[0] - np.argsort(variance) - 1  # reversed argsort
            self.selected_blocks = [blocks[i] for i in var_idx[:n]]
            return self.selected_blocks

        else:
            raise NotImplementedError
