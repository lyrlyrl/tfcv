from tqdm import tqdm
import os
import sys

__all__ = ['get_tqdm_kwargs', 'get_tqdm']

def _pick_tqdm_interval(file):
    # Heuristics to pick a update interval for progress bar that's nice-looking for users.
    isatty = file.isatty()
    # Jupyter notebook should be recognized as tty.
    # Wait for https://github.com/ipython/ipykernel/issues/268
    try:
        from ipykernel import iostream
        if isinstance(file, iostream.OutStream):
            isatty = True
    except ImportError:
        pass

    if isatty:
        return 0.5
    else:
        # When run under mpirun/slurm, isatty is always False.
        # Here we apply some hacky heuristics for slurm.
        if 'SLURM_JOB_ID' in os.environ:
            if int(os.environ.get('SLURM_JOB_NUM_NODES', 1)) > 1:
                # multi-machine job, probably not interactive
                return 60
            else:
                # possibly interactive, so let's be conservative
                return 15

        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            return 60

        # If not a tty, don't refresh progress bar that often
        return 180
        
def get_tqdm_kwargs(**kwargs):
    """
    Return default arguments to be used with tqdm.

    Args:
        kwargs: extra arguments to be used.
    Returns:
        dict:
    """
    default = dict(
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]'
    )

    try:
        # Use this env var to override the refresh interval setting
        interval = float(os.environ['TFCV_PROGRESS_REFRESH'])
    except KeyError:
        interval = _pick_tqdm_interval(kwargs.get('file', sys.stderr))

    default['mininterval'] = interval
    default.update(kwargs)
    return default

def get_tqdm(*args, **kwargs):
    """ Similar to :func:`tqdm.tqdm()`,
    but use mlco's default options to have consistent style. """
    return tqdm(*args, **get_tqdm_kwargs(**kwargs))