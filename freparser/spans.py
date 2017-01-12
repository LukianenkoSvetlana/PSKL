from collections import defaultdict

from .utils import parse_string


class Span:
    def __init__(self, id, type, sym_start, sym_len, tok_start, tok_len, tokens):
        self.id = id
        self.type = type
        self.sym_start = sym_start
        self.sym_len = sym_len
        self.tok_start = tok_start
        self.tok_len = tok_len
        self.tokens = tokens.slice(self.tok_start, self.tok_len)

    def __str__(self):
        return " ".join(str(tok) for tok in self.tokens)

    def __repr__(self):
        return "<Span: {} ({})>".format(self, self.id)

    @classmethod
    def load_from_buffer(cls, buf, tokens):
        id, type, sym_start, sym_len, tok_start, tok_len = parse_string("isiiii", buf)
        return cls(
            id=id,
            type=type,
            sym_start=sym_start,
            sym_len=sym_len,
            tok_start=tok_start,
            tok_len=tok_len,
            tokens=tokens,
        )


class SpansStorage:
    def __init__(self):
        self._id_index = {}
        self._tokens_index = {}
        self._types_index = defaultdict(list)
        self._all_ids = []

    def __getitem__(self, id):
        return self._id_index[id]

    def add_span(self, span):
        self._id_index[span.id] = span
        for token in span.tokens:
            self._tokens_index[token.id] = span.id
        self._types_index[span.type].append(span.id)
        self._all_ids.append(span.id)

    def all(self):
        return [
            self[id]
            for id in self._all_ids
        ]

    def get_by_token(self, token):
        token_id = token if type(token) is int else token.id
        span_id = self._tokens_index.get(token_id, None)
        if span_id is None:
            return None
        return self[span_id]

    def list_by_type(self, type):
        return [
            self[token_id]
            for token_id in self._types_index[type]
        ]

    @classmethod
    def load_from_file(cls, filename, tokens):
        result = cls()
        with open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                span = Span.load_from_buffer(line, tokens)
                result.add_span(span)
        return result
