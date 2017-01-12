from itertools import chain
from collections import defaultdict

from .utils import parse_string


class Fact:
    def __init__(self, id, type, props_pairs):
        self.id = id
        self.type = type
        self.props = {}
        self.verbose_props = defaultdict(list)
        for key, value in props_pairs:
            self.props[key] = value
            self.verbose_props[key].append(value)


    def __str__(self):
        return str(self.type)

    def __repr__(self):
        return "<Fact: {} ({})>".format(self, self.id)

    def get_value(self, key):
        return (
            self.props[key][0]
            if len(self.props[key]) > 0
            else None
        )

    def get_values_list(self, key):
        return self.props[key]

    @classmethod
    def load_from_buffer(cls, buf):
        lines = buf.strip().split("\n")
        first_line, props_lines = lines[0], lines[1:]
        id, type = parse_string("ss", first_line)
        props_triples = [
            line.strip().split(" ", 2)
            for line in props_lines
            if line.strip() != ""
        ]
        props_pairs = [
            (p[0], tuple(p[1:]), )
            for p in props_triples
        ]
        return cls(
            id=id,
            type=type,
            props_pairs=props_pairs,
        )


class FactsStorage:
    def __init__(self):
        self._id_index = {}
        self._types_index = defaultdict(list)
        self._all_ids = []

    def __getitem__(self, id):
        return self._id_index[id]

    def add_fact(self, fact):
        self._id_index[fact.id] = fact
        self._types_index[fact.type].append(fact.id)
        self._all_ids.append(fact.id)

    def all(self):
        return [
            self[id]
            for id in self._all_ids
        ]

    def list_by_type(self, type):
        return [
            self[token_id]
            for token_id in self._types_index[type]
        ]

    @classmethod
    def load_from_file(cls, filename):
        result = cls()
        with open(filename, "rt", encoding="utf-8") as f:
            buf = ""
            for line in chain(f, ""):
                if line.strip() == "":
                    if buf.strip() != "":
                        fact = Fact.load_from_buffer(buf)
                        result.add_fact(fact)
                        buf = ""
                else:
                    buf += line + "\n"
        return result
