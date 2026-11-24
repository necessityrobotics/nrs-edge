# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics
"""`python -m nrs_edge ...` is the same as the command `nrs-edge ...`."""
import sys

from .cli import main

sys.exit(main())
