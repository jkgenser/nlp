import typing as t
from datasets import Dataset, ClassLabel


class TextLabel:
    def __init__(self, start: int, end: int, label: str, text: str = None):
        start: int = start
        end: int = end
        label: str = label
        text: str = text


def overlaps(seq1: t.Tuple[int, int], seq2: t.Tuple[int, int]):
    """
    "two spans overlap if both of their starts come before both of their ends"
    """
    start = 0
    end = 1
    return seq1[start] <= seq2[end] and seq2[start] <= seq1[end]


def class_label_for_ds(ds: Dataset, label_col_name: str = "text_labels") -> ClassLabel:
    """
    For sequence labeling we assume that the labels are formatted like
    [{"start": int, "end": int, "label": str}]
    """
    class_set = set()
    for row in ds[label_col_name]:
        for label in row:
            class_set.add(label["label"])
    return ClassLabel(num_classes=len(class_set), names=list(class_set))
