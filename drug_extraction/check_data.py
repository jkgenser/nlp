import pandas as pd


ANN_FILE = "/home/jerry/nlp/data/training_20180910/100035.ann"
TEXT_FILE = "/home/jerry/nlp/data/training_20180910/100035.txt"

ann = open(ANN_FILE).read()
text = open(TEXT_FILE).read()


def handle_t(line: str):
    splitted = line.split()
    record = {
        "id": splitted[0],
        "type": splitted[1],
        "start": int(splitted[2]),
        "end": int(splitted[3]),
        "text": " ".join(splitted[4:]),
    }
    return record


def parse_line(line: str):
    if line == "":
        return

    # starts with T means that this is a
    # entity tag
    if line.startswith("T"):
        return handle_t(line)

    # We aren't parsing relation tags yet
    if line.startswith("R"):
        pass


def parse_ann_file(ann: str):
    records = []
    for line in ann.split("\n"):
        parsed = parse_line(line)
        if parsed:
            records.append(parsed)
    return records


parsed_ann = parse_ann_file(ann)


import ipdb

ipdb.set_trace()
