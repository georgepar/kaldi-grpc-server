import fileinput
import os


def to_sclite_line(trans):
    with open(trans, "r") as fd:
        hyp = fd.read()

    _id, _ = os.path.splitext(os.path.basename(trans))

    return f"{hyp} ({_id})"


def main():
    with fileinput.input() as finput:
        for ln in finput:
            print(to_sclite_line(ln.strip()))


if __name__ == "__main__":
    main()
