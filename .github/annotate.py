# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""Post a log file as one GitHub Actions annotation (annotations are public; job logs need a sign-in).

    python .github/annotate.py notice <title> <file>
    python .github/annotate.py error <title> <file>
"""
import sys


def esc(s):
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    kind, title, path = sys.argv[1], sys.argv[2], sys.argv[3]
    text = open(path, encoding="utf-8", errors="replace").read()
    title = esc(title).replace(":", "%3A").replace(",", "%2C")
    print("::%s title=%s::%s" % ("error" if kind == "error" else "notice", title, esc(text[:12000])))


if __name__ == "__main__":
    main()
