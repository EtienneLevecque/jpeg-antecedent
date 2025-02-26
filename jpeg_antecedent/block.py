import numpy as np
from heapq import heappop, heappush
import gurobipy as gp
from gurobipy import GRB

from jpeg_antecedent.pipeline import ComposedPipeline
import jpeg_antecedent.utils as utils


def send_logs(verbose, shared_dict, task_id, iteration, max_iteration, done):
    if verbose and shared_dict is not None and iteration % 100 == 0:
        tmp_dict = shared_dict[task_id]
        tmp_dict.update({'completed': iteration, 'total': max_iteration})
        shared_dict[task_id] = tmp_dict
    elif verbose and shared_dict is None and done:
        print(f"\rDONE {iteration}/{max_iteration}")
    elif verbose and shared_dict is None and iteration % 100 == 0:
        print(f"\rRUNNING {iteration}/{max_iteration}")


class Block:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.antecedents = dict()
        self.status = dict()
        self.iterations = dict()
        self.ignored = False
        self.is_clipped = dict()

    def search_antecedent(self,
                          pipeline: ComposedPipeline,
                          max_iter: int,
                          shared_dict: dict = None,
                          task_id: int = None,
                          verbose: bool = False,
                          rng: np.random.RandomState = None):
        """
        Search a dct antecedent to the spatial_block assuming it was obtained with the given pipeline.

        Args:
            pipeline: must have a forward_dct, an inverse_dct, a round_pixel and a round_dct method
            max_iter: maximum number of explorations. One exploration means expanding a parent into 128 possible
            children. The number of calls to inverse_dct is then 128 * max_iter
            shared_dict: a manager shared memory to track tasks status
            task_id: an integer to track the task status
            verbose: if true, the function will use the task_id and the shared_dict to communicate advancement
            rng: random number generator
        Returns:
            results: an antecedent if it has been found, None otherwise
            iter_count: iteration counter
        """
        if rng is None:
            rng = np.random.RandomState(123)
        iter_counter = 0
        c, n, m = self.value.shape
        target = np.array([self.value], dtype=np.int16)
        float_start = pipeline.backward(target)
        start = utils.round(float_start).astype(np.int16)

        if np.allclose(pipeline.forward(start), target):
            send_logs(verbose, shared_dict, task_id, max_iter, max_iter, True)
            self.status[pipeline] = 1
            self.antecedents[pipeline] = start
            self.iterations[pipeline] = 0
            return self.antecedents[pipeline]

        queue = [(np.zeros(0), 0., start.astype(np.int16).copy())]
        start.flags.writeable = False
        open_set = {start.data.tobytes().__hash__()}

        s = c * n * m  # (channel * row * column)
        upper_bound_offset = 0
        if self.is_clipped[pipeline]:
            # clipped blocks violate upper bound constraint, thus the upper bound is increased by an arbitrary factor of 5
            upper_bound_offset = 5 * pipeline.upper_bound

        mask = np.ravel(pipeline.pipelines[0].quant_tbl.astype(float) <= pipeline.upper_bound + upper_bound_offset)
        n_changes = np.sum(mask)
        changes = np.stack([np.eye(s, dtype=np.int8)[mask],
                            -np.eye(s, dtype=np.int8)[mask]]).reshape(2 * n_changes, c, n, m)
        changes_first_iteration = np.stack([np.eye(s, dtype=np.int8),
                                            -np.eye(s, dtype=np.int8)]).reshape(2 * s, c, n, m)
        not_ignored = np.zeros(2 * n_changes, dtype=bool)  # boolean mask
        children_hash = np.zeros(2 * n_changes, dtype=np.int64)  # temporary storage of hashes
        not_ignored_first_iteration = np.zeros(2 * s, dtype=bool)  # boolean mask
        children_hash_first_iteration = np.zeros(2 * s, dtype=np.int64)

        # First iteration uses every possible changes to avoid bad starting position without any solution
        send_logs(verbose, shared_dict, task_id, iter_counter, max_iter, False)

        _, _, current = heappop(queue)  # shape (1, 1 for grayscale or 3, 8, 8)
        children = current + changes_first_iteration  # all changes
        children.flags.writeable = False  # read-only array to hash it
        for i in range(2 * s):
            child = children[i]
            children_hash_first_iteration[i] = child.data.tobytes().__hash__()
            not_ignored_first_iteration[i] = children_hash_first_iteration[i] not in open_set
        children.flags.writeable = True
        distance = np.abs((children - float_start) * pipeline.pipelines[0].quant_tbl)
        norm_distance = np.linalg.norm(distance, axis=(2, 3), ord='fro')
        not_ignored_first_iteration = not_ignored_first_iteration & np.all(
            norm_distance <= np.ravel(pipeline.upper_bound + upper_bound_offset),
            axis=-1)

        if np.any(not_ignored_first_iteration):
            transformed_children = pipeline.forward(children[not_ignored_first_iteration])
            abs_error = np.abs(target - transformed_children)
            error = np.sum(abs_error, axis=(1, 2, 3))
            error_idx = np.argsort(error)

            if error[error_idx[0]] == 0:  # check only the first element which is the smallest error
                send_logs(verbose, shared_dict, task_id, max_iter, max_iter, True)

                self.antecedents[pipeline] = children[not_ignored_first_iteration][error_idx[0]]
                self.status[pipeline] = 1
                self.iterations[pipeline] = iter_counter

                return self.antecedents[pipeline]

            else:
                for idx in error_idx:
                    to_enqueue = (int(error[idx]), rng.random(),
                                  children[not_ignored_first_iteration][idx].copy())

                    heappush(queue, to_enqueue)
                    open_set.add(children_hash_first_iteration[not_ignored_first_iteration][idx])

        # Then for all remaining iterations, we use fewer changes depending on the upper bound of the pipeline
        for iter_counter in range(max_iter):

            if not queue:
                break

            send_logs(verbose, shared_dict, task_id, iter_counter, max_iter, False)

            _, _, current = heappop(queue)  # shape (1, 1 for grayscale or 3, 8, 8)
            children = current + changes
            children.flags.writeable = False  # read-only array to hash it
            for i in range(2 * n_changes):
                child = children[i]
                children_hash[i] = child.data.tobytes().__hash__()
                not_ignored[i] = children_hash[i] not in open_set
            children.flags.writeable = True
            distance = np.abs((children - float_start) * pipeline.pipelines[0].quant_tbl)
            norm_distance = np.linalg.norm(distance, axis=(2, 3), ord='fro')
            not_ignored = not_ignored & np.all(norm_distance <= np.ravel(pipeline.upper_bound + upper_bound_offset),
                                               axis=-1)

            if np.any(not_ignored):
                transformed_children = pipeline.forward(children[not_ignored])
                abs_error = np.abs(target - transformed_children)
                error = np.sum(abs_error, axis=(1, 2, 3))
                error_idx = np.argsort(error)

                if error[error_idx[0]] == 0:  # check only the first element which is the smallest error
                    send_logs(verbose, shared_dict, task_id, max_iter, max_iter, True)

                    self.antecedents[pipeline] = children[not_ignored][error_idx[0]]
                    self.status[pipeline] = 1
                    self.iterations[pipeline] = iter_counter

                    return self.antecedents[pipeline]

                else:
                    for idx in error_idx:
                        to_enqueue = (int(error[idx]), rng.random(),
                                      children[not_ignored][idx].copy())

                        heappush(queue, to_enqueue)
                        open_set.add(children_hash[not_ignored][idx])

        send_logs(verbose, shared_dict, task_id, max_iter, max_iter, True)
        self.iterations[pipeline] = iter_counter + 1
        if iter_counter + 1 < max_iter:
            self.antecedents[pipeline] = False
            self.status[pipeline] = -1
        else:
            self.antecedents[pipeline] = None
            self.status[pipeline] = 0
        return self.antecedents[pipeline]

    def search_antecedent_ilp(self,
                              parameters: dict,
                              shared_dict: dict = None,
                              task_id: int = None,
                              verbose: bool = False):
        """
        Search a spatial antecedent to the dct_block assuming it was obtained with the naive pipeline.

        This function uses Gurobi, a licensed ILP solver. You can get a free license with your institution here:
        https://www.gurobi.com/solutions/licensing/

        Args:
            parameters: a dictionary with Gurobi parameters
            shared_dict: a manager shared memory to track tasks status
            task_id: an integer to track the task status
            verbose: if True, the function will use the task_id and the shared_dict to communicate advancement

        Returns:
            results: an antecedent if it has been found, None otherwise
            iter_count: iteration counter
        """

        def my_callback(model, where):
            # TODO: verify this function when connected to the university network
            print(where, model.shared_dict, model.task_id)
            if where == GRB.Callback.MESSAGE:
                completed = model.cbGet(GRB.Callback.MIP_ITRCNT)
                send_logs(model.verbose, model.shared_dict, model.task_id, completed, m.max_iter, False)
            #    print(model.cbGet(GRB.Callback.SPX_OBJVAL))

        D = utils.dct_matrix(8, 8)
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.start()
        m = gp.Model("Inverse_JPEG_compression", env=env)
        m.shared_dict = shared_dict
        m.task_id = task_id
        m.max_iter = parameters["IterationLimit"]
        m.verbose = verbose

        offset = utils.round(D.T @ np.ravel(self.value)) - D.T @ np.ravel(self.value)

        r = m.addVar(vtype=GRB.CONTINUOUS, name='r', lb=0, ub=1 / 2 + 1e-6, obj=1.0, column=None)
        u = m.addMVar(shape=(64,), vtype=GRB.CONTINUOUS, name="u", lb=-1 / 2 - 1e-6, ub=1 / 2 + 1e-6)
        x = m.addMVar(shape=(64,), vtype=GRB.INTEGER, name="x", lb=-1, ub=1)

        m.addConstr(D @ (x - offset) == u, 'tmp')
        m.addGenConstrNorm(r, u, GRB.INFINITY, 'norm')

        m.setObjective(r, GRB.MINIMIZE)

        # Initialize parameters
        for k, v in parameters.items():
            if v is not None:
                m.setParam(k, v)

        # Solve
        m.optimize(my_callback)

        # Standard output
        solver_status = m.status
        if solver_status in [3, 6, 7, 9]:
            return None, m.IterCount
        else:
            return x.X.reshape(self.value.shape), m.IterCount
