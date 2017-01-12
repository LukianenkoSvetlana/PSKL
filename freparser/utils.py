def parse_string(pattern, string):
    chunks = string.split("#")[0].strip().split()
    multitail = pattern.endswith("+")
    if multitail:
        pattern = pattern[:-1]
    for idx, p in enumerate(pattern):
        transformer = {
            "i": int,
            "s": str,
        }[p]
        if multitail and idx == len(pattern) - 1:
            yield [
                transformer(x)
                for x in chunks[idx:]
            ]
        else:
            yield transformer(chunks[idx])
