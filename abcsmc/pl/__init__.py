from .plotting import plot_results
from .plotting import plot_perturbation_sample
from .plotting import plot_all_perturbation_sample
from .plotting import plot_posterior, plot_empirical_posterior
from .plotting import plot_kernel_comparison_results

def is_notebook() -> bool:
    """Check if current environment is a Jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    