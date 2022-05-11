from typing import Mapping

import numpy as np


def sequence_to_onehot(
    sequence: str,
    mapping: Mapping[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3},
    map_unknown_to_x: bool = False,
) -> np.ndarray:
    """Maps the sequence into a one-hot encoded matrix.

    Follows the interface in AlphaFold.

    Args:
      sequence:
        A sequence such as a sequence of nucleotides
      mapping (optional):
        A dictionary mapping possible sequence items (nucleotides) to integers, { ACGT -> 0123 } by default.
      map_unknown_to_x (optional):
        Items not in the mapping will be mapped to "X".
        If there is no "X" in the mapping, an error will be thrown.
        False by default.

    Returns:
      A numpy array of shape (seq_len, num_unique_items) with one-hot encoding of
      the sequence.

    Raises:
      ValueError:
        If the mapping doesn't contain values from 0 to
        num_unique_items - 1 without gaps.

    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_items-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for item_index, item in enumerate(sequence):
        if not map_unknown_to_x:
            item_id = mapping[item]
        else:
            if item.isalpha():
                item_id = mapping.get(item, mapping["X"])
            else:
                raise ValueError(f"Invalid character in the sequence: {item}")
        one_hot[item_index, item_id] = 1

    return one_hot
