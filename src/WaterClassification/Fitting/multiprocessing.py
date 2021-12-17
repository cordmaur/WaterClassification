import pathos.multiprocessing as mp


class PoolManager:
    def __init__(self, n_cpus=None, existing_pool=None):
        self.existing_pool = existing_pool
        self.n_cpus = n_cpus

    def __enter__(self):
        # The mapping function will depend if it will run single-core or multi-core
        if self.n_cpus is None:
            self.pool = None
        else:
            # if pool is None, create a new the processing pool
            if self.existing_pool is None:
                self.pool = mp.ProcessingPool(ncpus=self.n_cpus)
                self.pool.clear()
            # otherwise, use existing pool and adjust number of cpus
            else:
                self.pool = self.existing_pool
                self.pool.ncpus = self.n_cpus

            # set the mapping function to the pool.map

        return self

    def __exit__(self, tp, value, traceback):
        if (self.pool is not None) and (self.existing_pool is None):
            self.pool.close()

    def map(self, func, *args):
        if self.pool:
            return self.pool.map(func, *args)
        else:
            return list(map(func, *args))

