import numpy as np

# numpy madvises large arrays to transparent hugepages; on a host with fragmented memory every such
# allocation stalls in kernel compaction (native volume load 0.11 s -> 4.4 s, worse under load)
np._core.multiarray._set_madvise_hugepage(False)

__version__ = "0.1.0"
