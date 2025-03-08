# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import itertools
import ujson

from colbert.evaluation.loaders import load_collection
from colbert.infra.run import Run


class Collection:
    def __init__(self, path=None, data=None):
        self.path = path
        self.data = data or self._load_file(path)

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.data.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.data[item]

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.data)

    def _load_file(self, path):
        self.path = path
        return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        raise NotImplementedError()

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def save(self, new_path):
        assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)
            
            return f.name

    def enumerate(self, rank):
        for _, offset, passages in self.enumerate_batches(rank=rank):
            for idx, passage in enumerate(passages):
                yield (offset + idx, passage)

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.

class TranslateAbleCollection(Collection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.translate_dict = self.try_load_translate_dict()

        self.translate_dict_rev = {int(v): int(k) for k, v in self.translate_dict.items()}

    def try_load_translate_dict(self):
        """
        Whole codebase works with
        new_collection_pids -> default_collection_pids
        """
        # Path does not exist if loaded from data not from path
        if self.path is None:
            return

        assert self.path[-4:] == '.tsv', f"Only tsv files are supported, got {self.path}"

        # example:
        # collection.dev.small_50-25-25.tsv and
        # collection.dev.small_50-25-25.translate_dict.json
        trans_dict_path = self.path[:-4] + '.translate_dict.json'
        with open(trans_dict_path) as reader:
            loaded = ujson.load(reader)
            loaded = {int(k): int(v) for k, v in loaded.items()}
            return loaded

    def translate(self, pid):
        """
        Translates custom collection passage id to default collection pid.
        """
        return self.translate_dict[pid]

    def translate_rev(self, pid):
        return self.translate_dict_rev[pid]

    def get_translated_pid(self, pid):
        return self[self.translate_dict[pid]]

    def get_translated_pid_rev(self, pid):
        return self[self.translate_dict_rev[pid]]

    @classmethod
    def cast(cls, obj):
        if type(obj) is Collection:
            return cls(path=obj.path, data=obj.data)

        raise AssertionError(f"obj has type {type(obj)} which is not compatible with cast()")
