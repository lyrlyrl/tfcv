import hashlib

def _resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == 'sha256':
        return hashlib.sha256()

    if algorithm == 'auto' and file_hash is not None and len(file_hash) == 64:
        return hashlib.sha256()

    # This is used only for legacy purposes.
    return hashlib.md5()

def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    Example:
    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    Args:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        The file hash
    """
    if isinstance(algorithm, str):
        hasher = _resolve_hasher(algorithm)
    else:
        hasher = algorithm

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()
    
def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    assert algorithm in ['sha256', 'md5'], f'algorithm should be one of "sha256" and "md5", but got {algorithm}'
    hasher = _resolve_hasher(algorithm, file_hash)
    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

