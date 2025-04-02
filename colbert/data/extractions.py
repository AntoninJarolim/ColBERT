import numpy as np
import ujson
from colbert.infra import Run
import torch


def _load_file(path):
    data = {}
    with open(path) as reader:
        for line in reader:
            q_id, psg_id, _, *span_list = line.strip().split('\t')
            q_id, psg_id = int(q_id), int(psg_id)
            data[(q_id, psg_id)] = span_list

    return data


class Extractions:
    def __init__(self, path=None, data=None, nway=None):
        self.path = path
        self.nway = nway
        self.data = data or _load_file(path)

        self.q_to_d = {int(q_id): int(psg_id) for q_id, psg_id in self.data.keys()}

    @classmethod
    def cast(cls, obj, nway=None):
        if type(obj) is str:
            return cls(path=obj, nway=nway)

        if isinstance(obj, dict):
            return cls(data=obj, nway=nway)

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def get_psg_by_qid(self, qid: int) -> int:
        return self.q_to_d[qid]


class ExtractionResults:
    valid_data_keys = ['psg_id', 'q_id',
                       'extraction_binary', 'extraction_full',
                       'max_scores', 'max_scores_full']

    def __init__(self, data=None, metadata=None, skip_special=True):
        self.data = data
        self.skip_special = skip_special

        if skip_special:
            self.data = self.skip_special_data()

        self.metadata = {} if metadata is None else metadata
        assert type(self.metadata) is dict, f"metadata initialized with type {type(self.metadata)}"

    @classmethod
    def cast(cls, obj, **kwargs):
        if type(obj) is list:
            # Keep only the valid keys
            keep_list = [{key: member[key] for key in cls.valid_data_keys} for member in obj]

            # Convert tensors to lists
            for member in keep_list:
                for key in cls.valid_data_keys:
                    if type(member[key]) is torch.Tensor:
                        member[key] = member[key].tolist()

            # check that extraction_binary', 'max_scores' lens are the same
            extractions_only = [member['extraction_binary'] for member in keep_list]
            max_scores_only = [member['max_scores'] for member in keep_list]
            assert all([len(x) == len(y) for x, y in zip(extractions_only, max_scores_only)])

            return cls(data=keep_list, **kwargs)

        if type(obj) is str:
            data, metadata = cls.open_jsonl(obj)
            return cls(data=data, metadata=metadata, **kwargs)

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

    def save(self, new_path):
        assert new_path.split('.')[-1] == 'jsonl'
        with Run().open(new_path, 'w', save_dir='results') as f_data:
            for d in self.data:
                f_data.write(ujson.dumps(d) + "\n")

        with Run().open(f'{new_path}.meta', 'w', save_dir='results') as f:
            assert type(self.metadata) is dict
            meta = {
                'metadata:': self.metadata,
            }
            ujson.dump(meta, f, indent=4)
        return f_data.name

    @classmethod
    def open_jsonl(cls, path):
        data = []
        with open(path) as reader:
            for line in reader:
                data.append(ujson.loads(line.strip()))

        path_meta = path + '.meta'
        with open(path_meta) as reader:
            metadata = ujson.load(reader)

        return data, metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def skip_special_data(self):
        keys_remove_special = [
            'extraction_binary',
            'extraction_full',
            'max_scores',
            'max_scores_full',
        ]

        new_data = []
        for d in self.data:

            for k in ['extraction_binary', 'extraction_full']:
                assert np.all(np.array(d[k][:2]) == 0), 'Removing non-zero scores'
                assert np.all(np.array(d[k][-1]) == 0), 'Removing non-zero scores'

            for k in keys_remove_special:
                # Remove first two -> [CLS] [D]
                # Remove last one -> [SEP]
                d[k] = d[k][2:-1]
            new_data.append(d)

        return new_data
