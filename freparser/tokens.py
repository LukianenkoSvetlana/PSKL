from collections import defaultdict

from .utils import parse_string


class Token:
    def __init__(self, id, start_pos, length, text, sentence_idx):
        self.id = id
        self.start_pos = start_pos
        self.length = length
        self.text = text
        self.sentence_idx = sentence_idx

    def __str__(self):
        return self.text

    def __repr__(self):
        return "<Token: {} ({})>".format(self, self.id)

    @classmethod
    def load_from_buffer(cls, buf, sentence_idx):
        id, spos, length, text = parse_string("iiis", buf)
        return cls(
            id=id,
            start_pos=spos,
            length=length,
            text=text,
            sentence_idx=sentence_idx,
        )


class TokensStorage:
    def __init__(self):
        self._sequence = []
        self._id_to_seq_index = {}
        self._id_index = {}
        self._sentences = defaultdict(list)

    def __getitem__(self, id):
        return self._id_index[id]

    def add_token(self, token):
        self._id_index[token.id] = token
        self._sentences[token.sentence_idx].append(token.id)
        seq_pos = len(self._sequence)
        self._sequence.append(token.id)
        self._id_to_seq_index[token.id] = seq_pos

    def get_sentence(self, idx):
        return [
            self[token_id]
            for token_id in self._sentences[idx]
        ]

    @property
    def sentences(self):
        for key in sorted(self._sentences.keys()):
            yield self.get_sentence(key)

    def slice(self, first_id, size):
        seq_pos = self._id_to_seq_index[first_id]
        return [
            self[id]
            for id in self._sequence[seq_pos:seq_pos + size]
        ]

    def slice_by_ids(self, first_id, last_id):
        first_seq_pos = self._id_to_seq_index[first_id]
        last_seq_pos = self._id_to_seq_index[last_id]
        return [
            self[id]
            for id in self._sequence[first_seq_pos:last_seq_pos]
        ]

    @classmethod
    def load_from_file(cls, filename):
        result = cls()
        with open(filename, "rt", encoding="utf-8") as f:
            sentence_idx = 0
            for line in f:
                if line.strip() == "":
                    sentence_idx += 1
                    continue
                token = Token.load_from_buffer(line, sentence_idx)
                result.add_token(token)
        return result
