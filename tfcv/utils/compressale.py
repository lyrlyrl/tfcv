import tarfile
import zipfile
import os
import shutil

__all__ = ["extract_archive", "compress_archive"]

def path_to_string(path):
    if isinstance(path, os.PathLike):
      return os.fspath(path)
    return path

def extract_archive(file_path, path='.', archive_format='auto'):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Arguments:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = path_to_string(file_path)
    path = path_to_string(path)

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        elif archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        else:
            raise ValueError("not valid archive_type")
        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False

def compress_archive(file_name, dirname, archive_format='auto'):
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar']
    if isinstance(archive_format, str):
        archive_format = [archive_format]
    for archive_type in archive_format:
        if archive_type == 'tar':
            file_name=file_name+".tar.gz"
            if os.path.isfile(dirname):
                with tarfile.open(file_name, 'w:gz') as tar:
                    tar.add(dirname, arcname=dirname.split('/')[-1])
            else:
                dr=dirname.split('/')[-1]
                with tarfile.open(file_name, 'w:gz') as tar:
                    for root, dirs, files in os.walk(dirname):
                        for single_file in files:
                            # if single_file != tarfilename:
                            filepath = os.path.join(root, single_file)
                            tar.add(filepath, arcname=filepath.replace(dirname, dr))
        elif archive_type == 'zip':
            file_name=file_name+".zip"
            if os.path.isfile(dirname):
                with zipfile.ZipFile(file_name, "w", zipfile.ZIP_DEFLATED) as f:
                    f.write(dirname)
            else:
                with zipfile.ZipFile(file_name, "w", zipfile.ZIP_DEFLATED) as f:
                    for root, dirs, files in os.walk(dirname):
                        for single_file in files:
                            # if single_file != tarfilename:
                            filepath = os.path.join(root, single_file)
                            f.write(filepath)
        else:
            raise ValueError("not valid archive_type")
    return file_name