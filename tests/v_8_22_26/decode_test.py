import os
import mmap
import struct
from datetime import datetime, timezone
import pandas as pd
import re

file = "tests/v_8_22_26/nda/nda_v8_2.nda"

with open(file, "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

print(mm.read(6))