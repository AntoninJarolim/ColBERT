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

    @classmethod
    def cast(cls, obj, nway=None):
        if type(obj) is str:
            return cls(path=obj, nway=nway)

        if isinstance(obj, dict):
            return cls(data=obj, nway=nway)

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"

    def __getitem__(self, key):
        return self.data[key]