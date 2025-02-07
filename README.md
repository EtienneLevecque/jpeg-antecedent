> [!WARNING]  
> This package is still under active development. The structure and some functions may change substantially.

# JPEG Antecedent

Repository linked with the two following papers:
- [\[1\] Finding Incompatible Blocks for Reliable JPEG Steganalysis](https://arxiv.org/pdf/2402.13660.pdf) (E. Levecque,
J. Butora, P. Bas)
-  [\[2\] Dual JPEG Compatibility: a Reliable and Explainable Tool for Image Forensics](https://arxiv.org/pdf/2408.17106v1) 
(E. Levecque, J. Butora, P. Bas)

In our first work [\[1\]](https://arxiv.org/pdf/2402.13660.pdf "Finding Incompatible Blocks for Reliable JPEG Steganalysis") we have shown that JPEG compression can be seen as a mathematical function that maps a pixel 
block to a DCT block. But if you modify your DCT block, does it still have a pixel antecedent? If the answer is no, then
this block is incompatible!
This work also shows that most of the time at QF100, the bigger the modification, the more likely your block will be
incompatible. This method can reliably detect steganography messages embedded in an image at QF100 and outperform the 
other methods, especially for very small messages.

In the second contribution [\[2\]](https://arxiv.org/pdf/2408.17106v1 "Dual JPEG Compatibility: a Reliable and Explainable Tool for Image Forensics"), we extend the search to antecedents of pixel block to have both the search for the 
compression and the decompression. These two elements can be combined to search for an antecedent of any chained pipeline 
of alternated compression and decompression. This is called the Dual JPEG Antecedent Search algorithm.
In particular, this dual formulation can be used to detect double compressed JPEG blocks with perfect accuracy and thus
detect manipulated JPEG images even after a second JPEG compression.

## Installation

Require **Python >= 3.12**.

Install the package:

    pip install jpeg-antecedent

## Examples

If you have an image, and you want to play with it:

```python
from jpeg_antecedent.image import Image
from jpeg_antecedent.pipeline import create_pipeline

path = r"path/to/my/image"

img = Image('toy_image', path)
pipeline = create_pipeline('islow', 100, img.is_grayscale, target_is_dct=img.is_jpeg)  # Islow pipeline with QF100

# Example for a double compression pipeline:
# pipeline = create_pipeline(['islow', 'naive'], [90,100], img.is_grayscale, target_is_dct=img.is_jpeg) 

img.set_pipeline(pipeline)
img.set_selection_parameter('random', 10)  # 10% of the blocks randomly

img.search_antecedent(1000)

for pos, block in img.block_collection.items():
    if pipeline in block.status:
        # status is:
        # 3 if block has been ignored
        # 1 if an antecedent has been found
        # 0 otherwise (potentially incompatible)
        # -1 for incompatible
        status = block.status[pipeline]
        antecedent = block.antecedents[pipeline]
        iteration = block.iterations[pipeline]
```

If you have a single block:

```python
import numpy as np
from jpeg_antecedent.block import Block
from jpeg_antecedent.pipeline import create_pipeline

array = np.random.randint(-1016, 1016, size=(8, 8))  # random DCT block
block = Block(array)
pipeline = create_pipeline('float', 90, grayscale=True, target_is_dct=True)  # Float pipeline with QF90 for grayscale blocks

block.search_antecedent(pipeline, 1000)
antecedent = block.antecedents[pipeline]
iteration = block.iterations[pipeline]
if antecedent is None:
    print('No antecedent found, could be incompatible.')
else:
    print(f"Compatible!\nFound an antecedent in {iteration} iterations: \n{antecedent}")
```

## Dataset usage (to be updated)

If you have a dataset of images or want to customize your search, please use the [config file](config.yaml) as follows.

    experiment_name: Experiment Name

    seed: 123

Every experiment with the same name will be stored in the same directory. Seed is for reproducibility.

    data:
      input_path: "path/to/my/image/folder"
      output_path: "path/to/my/output/folder"
      starting_index: 0 # start at the first image of the folder
      ending_index: -1 # end at the last image of the folder

``starting_index`` and ``ending_index`` can be used to select a subset of images. For example, if you have 50 images in
your folder, and you set ``starting_index=45`` and `ending_index=-1`. It will only work with the 5 last images in ascii
filename order.

      preprocessing:
        avoid_clipped: True
        avoid_uniform: True
        percentage_block_per_image: 100
        sorting_method: variance

Parameters to filter out clipped or uniform blocks (recommended) but also to select a subset of blocks in the image.
Most of the time, 10% of the blocks are sufficient to classify the image as modified or not.
See [this section](#blocks-selection-and-filtering) for more details.

    antecedent_search:
      max_workers: null # int, null to use all available CPUs
      pipeline: naive
      quant_tbl: 100

Definition of the expected pipeline for the image. See the [Pipeline](#Pipeline) section for more details.

      heuristic_solver:
        use: True
        max_iteration_per_block: 1000

The heuristic solver is a local search to find antecedents. It is quite fast but cannot prove that a block is
incompatible.

      gurobi_solver: # !! only possible if pipeline == naive !!
        use: False
        max_iteration_per_block: 1000
        mip_focus: 1 # 1, 2 or 3
        threads: 3 # !! total number of jobs will be threads * worker !!
        cutoff: 0.500001
        node_file_start: 0.5 # null for no RAM usage limit

Gurobi is a licensed ILP solver but free licenses are available with a public institution. This solver can only be used
for JPEG images at QF100 with a naive pipeline.

With your custom config file, run the following command :

    python3 -m antecedent.run_on_dataset <your_config.yaml> --verbose


## Pipeline

There are currently 4 pipelines implemented:

|    Class name     | parameter name | Description                                                                                                                                                     |
|:-----------------:|:--------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ``NaivePipeline`` |   ``naive``    | Mathematical DCT using the [scipy.fft.dctn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html). Most precise but rarely used for images. |
| ``IslowPipeline`` |   ``islow``    | [Libjpeg islow](https://github.com/winlibs/libjpeg/blob/master/jfdctint.c). Uses a 32 bits integer DCT algorithm.                                               |
| ``FloatPipeline`` |   ``float``    | [Libjpeg float](https://github.com/winlibs/libjpeg/blob/master/jfdctflt.c). Uses a floating-point DCT algorithm.                                                |
| ``IfastPipeline`` |   ``ifast``    | [Libjpeg ifast](https://github.com/winlibs/libjpeg/blob/master/jfdctfst.c). Uses a 16 bits integer DCT algorithm.                                               |

You can add your custom pipeline to use different DCT algorithms or rounding functions. To do so, modify
the [pipeline.py](antecedent/pipeline.py) with a new class with the following methods:

```python
from jpeg_antecedent.pipeline import JPEGPipeline


class YourCustomPipeline(JPEGPipeline):
    def __init__(self, quality, grayscale, target_is_dct):
        super().__init__('your_custom_name', quality, grayscale, target_is_dct)

    @classmethod
    def is_named(cls, name):
        return name == 'your_custom_name'

    # Necessary to run the code
    def dct(self, blocks):
        pass

    def idct(self, blocks):
        pass

    # Not necessary but available to custom your pipeline
    def round_dct(self, blocks):
        pass

    def round_pixel(self, blocks):
        pass

    def rgb_to_ycc(self, blocks):
        pass

    def ycc_to_rgb(self, blocks):
        pass

    def define_quant_table(self):
        # Must define self.quant_tbl
        pass
```

Some useful functions can be found in [jpeg_toolbox.py](antecedent/jpeg_toolbox.py).

### Double compression pipeline

Suppose you observe some JPEG images C and want to find antecedents A through a double compression pipeline:

    A (DCT) ---- pipe 1 ----> B (Pixel) ---- pipe 2 ----> C (DCT)

with:

- pipe 1: Naive pipeline with QF75
- pipe 2: Islow pipeline with QF98

You can define the pipelines in the config file as follows:

    antecedent_search:
      max_workers: null # int, null to use all available CPUs
      pipeline: [naive, islow]
      quant_tbl:[75, 98]

Or directly in python as follows:

```python
from jpeg_antecedent.pipeline import create_pipeline

grayscale = True  # depends on your image/block

pipeline = create_pipeline(['naive', 'islow'], [75, 98], grayscale=grayscale, target_is_dct=True)
```

## Blocks selection and filtering

If you have a JPEG image at QF100 and you want to know if it has been modified, we have shown in our paper that we can
select blocks that are more likely to be incompatible by using the variance of the rounding error.
This selection will reduce the number of antecedent searches that are costly in time and keep very good (sometimes
better) results.

| parameter name | Description                                                                          |
|:--------------:|--------------------------------------------------------------------------------------|
|   ``random``   | Random selection                                                                     |
|  ``variance``  | Select blocks with the highest spatial rounding error variance (Not implemented yet) |
|    ``pmap``    | Select blocks with the highest probability of modification (Not implemented yet)     |

## References

If you use our work, please cite us with one of the following citations:

```
    @article{levecque2024finding,
        author={Levecque, Etienne and Butora, Jan and Bas, Patrick},
        journal={IEEE Transactions on Information Forensics and Security}, 
        title={Finding Incompatible Blocks for Reliable JPEG Steganalysis}, 
        year={2024},
        volume={19},
        pages={9467-9479},
        doi={10.1109/TIFS.2024.3470650}
    }
```
```
    @preprint{levecque2024dual,
        title={Dual JPEG Compatibility: a Reliable and Explainable Tool for Image Forensics}, 
        author={Etienne Levecque and Jan Butora and Patrick Bas},
        year={2024},
        eprint={2408.17106},
        archivePrefix={arXiv},
        primaryClass={cs.CR},
        url={https://arxiv.org/abs/2408.17106}, 
    }
```

## License

This work is under [MIT License](LICENSE.md).