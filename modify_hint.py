#! /usr/bin/env python3

import sys
import pickle

with open(sys.argv[1], "rb") as f:
    P = pickle.load(f)

print(f"Writing to hint for problem {P['Name']}.")
print(P["Hint"])
print("Write new hint and press Ctrl-D when finished.")

from contextlib import suppress


def input_lines(prompt=None):
    with suppress(EOFError):
        if prompt is not None:
            yield input(prompt)
        yield from iter(input, None)

hint = input_lines()
P["Hint"] = "\n".join(list(hint))


with open(sys.argv[1], "wb") as f:
    pickle.dump(P, f)
