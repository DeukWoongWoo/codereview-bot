from typing import List


def split_patch(patch: str, max_size: int) -> List[str]:
    """Split a patch string into chunks not exceeding ``max_size`` characters.

    Parameters
    ----------
    patch: str
        Full patch string.
    max_size: int
        Maximum number of characters per chunk.

    Returns
    -------
    List[str]
        List of patch chunks.
    """
    if max_size <= 0:
        return [patch]

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for line in patch.splitlines(keepends=True):
        line_len = len(line)
        if length + line_len > max_size and current:
            chunks.append("".join(current))
            current = []
            length = 0
        current.append(line)
        length += line_len
    if current:
        chunks.append("".join(current))
    return chunks

