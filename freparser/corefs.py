from collections import defaultdict
from itertools import chain

from .utils import parse_string


class Coref:
    def __init__(self, id, obj_ids, props_pairs, objects):
        self.id = id
        self.objects = [objects[obj_id] for obj_id in obj_ids]
        self.props = {}
        self.verbose_props = defaultdict(list)
        self.type = self.objects[0].type
        for p in props_pairs:
            key = p[0]
            value = None if len(p) == 1 else p[1]
            self.props[key] = value
            self.verbose_props[key].append(value)

    def __str__(self):
        return str(self.objects[0])

    def __repr__(self):
        return "<Coref: {} ({})>".format(self, self.id)

    @classmethod
    def load_from_buffer(cls, buf, objects):
        lines = buf.strip().split("\n")
        first_line, props_lines = lines[0], lines[1:]
        id, obj_ids = parse_string("ii+", first_line)
        props_pairs = [
            line.strip().split(" ", 1)
            for line in props_lines
            if line.strip() != ""
        ]
        return cls(
            id=id,
            obj_ids=obj_ids,
            props_pairs=props_pairs,
            objects=objects,
        )


class CorefsStorage:
    def __init__(self):
        self._id_index = {}
        self._objects_index = {}
        self._all_ids = []

    def __getitem__(self, id):
        return self._id_index[id]

    def add_coref(self, coref):
        self._id_index[coref.id] = coref
        for obj in coref.objects:
            self._objects_index[obj.id] = coref.id
        self._all_ids.append(coref.id)

    def all(self):
        return [
            self[id]
            for id in self._all_ids
        ]

    def get_by_object(self, obj):
        obj_id = obj if type(obj) is int else obj.id
        id = self._objects_index.get(obj_id, None)
        if id is None:
            return None
        return self[id]

    @classmethod
    def load_from_file(cls, filename, objects):
        result = cls()
        with open(filename, "rt", encoding="utf-8") as f:
            buf = ""
            for line in chain(f, ""):
                if line.strip() == "":
                    if buf.strip() != "":
                        coref = Coref.load_from_buffer(buf, objects)
                        result.add_coref(coref)
                        buf = ""
                else:
                    buf += line + "\n"
        return result
