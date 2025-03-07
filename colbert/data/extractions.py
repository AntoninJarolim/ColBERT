import ujson
from colbert.infra import Run


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
    valid_data_keys = ['extraction_binary', 'max_scores', 'psg_id', 'q_id']

    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata

    @classmethod
    def cast(cls, obj):
        if type(obj) is list:
            keep_list = [{key: member[key] for key in cls.valid_data_keys} for member in obj]
            # todo: check that extraction_binary', 'max_scores' lens are the same
            extractions_only = [member['extraction_binary'] for member in keep_list]
            max_scores_only = [member['max_scores'] for member in keep_list]
            assert all([len(x) == len(y) for x, y in zip(extractions_only, max_scores_only)])
            return cls(data=keep_list)

        if type(obj) is str:
            raise NotImplementedError("Loading from file not supported yet.")

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
