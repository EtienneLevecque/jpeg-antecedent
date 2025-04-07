import os
import csv
import numpy as np
import yaml
import shutil
import argparse
import multiprocessing
import concurrent.futures
import pandas as pd

from time import strftime, localtime
from rich import progress
from PIL import Image as pimg

from jpeg_antecedent.image import Image
from jpeg_antecedent.pipeline import create_pipeline


def main(config, config_path, verbose=False):
    validate_config(config)
    path_to_csv, image_mask_output_dir, npy_mask_output_dir = create_output_folder(config, config_path, verbose)
    input_path = config['data']['input_path']

    filenames = sorted(os.listdir(input_path))
    start, stop = config['data']['starting_index'], config['data']['ending_index']
    if stop == -1:
        stop = len(filenames)

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    output_queue = manager.Queue()
    writing_process = multiprocessing.Process(target=to_csv, args=(path_to_csv, output_queue), daemon=True)
    writing_process.start()

    futures = []
    with progress.Progress(
            "[progress.description]{task.description}",
            progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            progress.TimeRemainingColumn(),
            progress.TimeElapsedColumn(),
            refresh_per_second=10) as progress_bar:

        with concurrent.futures.ProcessPoolExecutor(max_workers=config['antecedent_search']['max_workers']) as executor:
            for filename in filenames[start:stop]:
                task_id = progress_bar.add_task(f"{filename}", visible=False, start=False)
                shared_dict[task_id] = {"filename": filename, "current_block": 1, "total_block": 1, "completed": 0,
                                        "total": 0, "started": False}
                futures.append(executor.submit(handle_image,
                                               filename,
                                               config,
                                               shared_dict,
                                               output_queue,
                                               task_id,
                                               verbose))

            overall_task = progress_bar.add_task("Overall", visible=True)
            while sum([future.done() for future in futures]) < len(futures):
                update_bar(progress_bar, futures, overall_task, shared_dict)
            update_bar(progress_bar, futures, overall_task, shared_dict)

    output_queue.put(None)
    writing_process.join()

    get_image_from_output(path_to_csv, image_mask_output_dir, npy_mask_output_dir, input_path)


def update_bar(bar, futures, overall_task, shared_dict):
    n_finished = sum([future.done() for future in futures])
    bar.update(overall_task, completed=n_finished, total=len(futures))
    for task_id, update_data in shared_dict.items():
        completed = update_data["completed"]
        total = update_data["total"]
        current_block = update_data["current_block"]
        total_block = update_data["total_block"]
        filename = update_data["filename"]
        started = update_data["started"]
        # update the progress bar for this task:
        if started and bar.tasks[task_id].started == False:
            bar.start_task(task_id)
        bar.update(task_id,
                   description=f"{filename} {current_block}/{total_block}",
                   completed=completed,
                   total=total,
                   visible=(current_block < total_block) or (completed < total))


def handle_image(filename, config, shared_dict, output_queue, task_id, verbose):
    data = config['data']
    seed = config['seed']
    search_config = config['antecedent_search']
    preprocessing = data['preprocessing']
    max_iter_heuristic = search_config['heuristic_solver']['max_iteration_per_block']

    rng = np.random.RandomState(seed)

    img = Image(filename, os.path.join(data['input_path'], filename))

    if img.img is None:
        return None

    pipeline = create_pipeline(name=search_config['pipeline'],
                               quality=search_config['quality'],
                               grayscale=img.is_grayscale,
                               target_is_dct=img.is_jpeg)
    img.set_pipeline(pipeline)
    img.set_filter_parameter(preprocessing['avoid_clipped'])
    img.set_selection_parameter(preprocessing['sorting_method'], preprocessing['percentage_block_per_image'], rng=rng)

    if search_config['heuristic_solver']['use']:
        img.search_antecedent(max_iter_heuristic, shared_dict, task_id, verbose)

    if search_config['gurobi_solver']['use']:
        img.search_antecedent_ilp(search_config['gurobi_solver'], shared_dict, task_id, verbose)
    output_queue.put(img)


def to_csv(path_to_csv, output_queue):
    """
    Write data to a csv file.

    Args:
        path_to_csv: path to the output.csv file
        output_queue: queue in which the main algorithm outputs the images done and that need to be written to csv
    """
    while True:
        queue_element = output_queue.get(block=True)
        if queue_element is not None:
            image = queue_element
        else:
            return
        filename = image.filename
        pipeline = image.pipeline
        with open(path_to_csv, 'a') as f:
            writer = csv.writer(f)

            for pos, block in image.block_collection.items():
                is_clipped = block.is_clipped[pipeline]
                iteration_gurobi = 0  # FIXME
                if pipeline in block.status:
                    status = block.status[pipeline]
                    antecedent = block.antecedents[pipeline]
                    iteration_heuristic = block.iterations[pipeline]
                    if antecedent is not None:
                        sol = str(np.ravel(antecedent).astype(int).tolist())
                    else:
                        sol = None
                else:
                    sol = None
                    iteration_heuristic = 0
                    status = 3

                writer.writerow(
                    [filename, str(pos), str(pipeline), str(status), str(is_clipped), str(iteration_heuristic),
                     str(iteration_gurobi), sol])


def validate_config(config):
    data = config['data']
    antecedent_search = config['antecedent_search']
    if not os.access(data['input_path'], os.R_OK):
        raise PermissionError(f"{data['input_path']} is not readable.")
    if data['ending_index'] != -1 and data['starting_index'] > data['ending_index']:
        raise ValueError(f"starting_index must be lower than ending_index.")
    if data['preprocessing']['percentage_block_per_image'] <= 0:
        raise ValueError("The percentage of block used per image must be strictly positive")
    # TODO: improve pipeline verification
    # create_pipeline(antecedent_search['pipeline'], antecedent_search['quality'], True)


def create_output_folder(config, config_path, verbose):
    # Building output directory, files and logs
    # in the output path, here is the hierarchy:
    # output_path /
    # ├─ experiment_name /
    # │  ├─ job_id_date / (or just date)
    # │  │  ├─ config.yaml
    # │  │  ├─ outputs.csv
    # │  │  ├─ logs.txt

    experiment_name = config['experiment_name']
    data = config['data']
    if experiment_name is None:
        experiment_name = ''
    experiment_name = experiment_name.replace(' ', '_').lower()

    exp_dir = os.path.join(data['output_path'], experiment_name)
    output_dir_name = strftime("%d-%m-%y_%H-%M", localtime())

    output_dir = os.path.join(exp_dir, output_dir_name)
    output_file = os.path.join(output_dir, "outputs.csv")
    image_mask_output_dir = os.path.join(exp_dir, output_dir_name, 'image_mask')
    npy_mask_output_dir = os.path.join(exp_dir, output_dir_name, 'npy_mask')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_mask_output_dir, exist_ok=True)
    os.makedirs(npy_mask_output_dir, exist_ok=True)
    shutil.copy2(config_path.name, os.path.join(output_dir, "config.yaml"))

    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ['filename', 'pos', 'pipeline', 'status', 'is_clipped', 'iteration_heuristic', 'iteration_gurobi',
             'solution'])
    if verbose:
        print(f'Output folder successfully created at: \n{output_dir}')
    return output_file, image_mask_output_dir, npy_mask_output_dir


def parse_args():
    """
    Read arguments anc check the configuration files with constraints defined in the schema_config.py file.

    Return:
        config: dict or parameters
        verbose: bool

    Raise:
        YAMLError if the config file cannot be safely loaded.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',
                        type=argparse.FileType(mode='r'))
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--job_id', '-j')
    args = parser.parse_args()

    try:
        config = yaml.safe_load(args.config_file)
    except yaml.YAMLError as exc:
        raise exc

    return config, args.verbose, args.config_file, args.job_id


def get_image_from_output(path_to_csv, image_mask_output_dir, npy_mask_output_dir, input_path):
    df = pd.read_csv(path_to_csv)
    for filename, group in df.groupby('filename'):
        with pimg.open(os.path.join(input_path, filename)) as img:
            n, m = img.height, img.width
        mask = np.zeros((n, m), dtype=np.uint8)
        for i in range(group.shape[0]):
            pos = eval(group.iloc[i]['pos'])
            status = int(group.iloc[i]['status'])
            is_clipped = bool(group.iloc[i]['is_clipped'])

            if status == -1 and not is_clipped:
                mask[pos[0] * 8: (pos[0] + 1) * 8, pos[1] * 8: (pos[1] + 1) * 8] = 255
            elif status == 0 or (status == -1 and is_clipped):
                mask[pos[0] * 8: (pos[0] + 1) * 8, pos[1] * 8: (pos[1] + 1) * 8] = 128
            elif status == 3:
                mask[pos[0] * 8: (pos[0] + 1) * 8, pos[1] * 8: (pos[1] + 1) * 8] = 0
        pimg.fromarray(mask).save(os.path.join(image_mask_output_dir, filename) + '.png')
        np.save(os.path.join(npy_mask_output_dir, filename), mask)


if __name__ == "__main__":
    config, verbose, config_path, _ = parse_args()
    main(config, config_path, verbose)
