import os
import logging
import urllib
from urllib.request import urlretrieve

from tfcv.utils.compressale import extract_archive, path_to_string
from tfcv.utils.hashable import validate_file
from tfcv.utils.progress import get_tqdm

class DownloadProgbar:
    def __init__(self,
                total_size):
        self.bar = get_tqdm(
            desc='Download process',
            total=total_size,
            unit='B',
            unit_scale=True)
        self._bid = 0
    def update(self, count, block_size):
        self.bar.update((count - self._bid) * block_size)
        self._bid = count

def get_file(fname,
            origin,
            extract=False,
            file_hash=None,
            cache_subdir='download',
            hash_algorithm='auto',
            archive_format='auto',
            cache_dir=None):
    if cache_dir is None:
        cache_dir = '~/.mlco'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.mlco')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fname = path_to_string(fname)

    fpath = os.path.join(datadir, fname)
    if extract:
        fpath = os.path.join(datadir, fname)
        if archive_format == 'zip':
            fpath = fpath + '.zip'
        elif archive_format == 'auto':
            if origin.endswith('.zip'):
                fpath = fpath + '.zip'
            elif origin.endswith('.tar.gz'):
                fpath = fpath + '.tar.gz'
            else:
                pass
        else:
            fpath = fpath + '.tar.gz'

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                    'incomplete or outdated because the ' + hash_algorithm +
                    ' file hash does not match the original value of ' + file_hash +
                    ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        logging.info('Downloading data from {}'.format(origin))

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None


        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = DownloadProgbar(total_size)
            else:
                ProgressTracker.progbar.update(count, block_size)


        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if extract:
        extract_archive(fpath, datadir, archive_format)

    return fpath