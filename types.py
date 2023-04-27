import os
from typing import BinaryIO, Union, IO

from typing_extensions import TypeAlias  # Python 3.10+

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
