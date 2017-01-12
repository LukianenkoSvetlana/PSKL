from collections import defaultdict

from .utils import parse_string


class Object:
    def __init__(self, id, type, related_spans_ids, spans):
        self.id = id
        self.type = type
        self.related_spans = [spans[span_id] for span_id in related_spans_ids]

    def __str__(self):
        return str(self.related_spans[0])

    def __repr__(self):
        return "<Object: {} ({})>".format(self, self.id)

    @classmethod
    def load_from_buffer(cls, buf, spans):
        id, type, related_spans_ids = parse_string("isi+", buf)
        return cls(
            id=id,
            type=type,
            related_spans_ids=related_spans_ids,
            spans=spans,
        )


class ObjectsStorage:
    def __init__(self):
        self._id_index = {}
        self._spans_index = {}
        self._types_index = defaultdict(list)
        self._all_ids = []

    def __getitem__(self, id):
        return self._id_index[id]

    def add_object(self, obj):
        self._id_index[obj.id] = obj
        for span in obj.related_spans:
            self._spans_index[span.id] = obj.id
        self._types_index[obj.type].append(obj.id)
        self._all_ids.append(obj.id)

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

    def get_by_span(self, span):
        span_id = span if type(span) is int else span.id
        id = self._spans_index.get(span_id, None)
        if id is None:
            return None
        return self[id]

    @classmethod
    def load_from_file(cls, filename, spans):
        result = cls()
        with open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                obj = Object.load_from_buffer(line, spans)
                result.add_object(obj)
        return result
