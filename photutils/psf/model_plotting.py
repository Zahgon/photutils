
import numpy as np
from astropy.visualization import simple_norm

__all__ = []


def _plot_grid_docstring(func):
    func.__doc__ = """
        Plot the grid of ePSF models.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        vmax_scale : float, optional
            Scale factor to apply to the image stretch limits. This
            value is multiplied by the peak ePSF value to determine the
            plotting ``vmax``. The defaults are 1.0 for plotting the
            ePSF data and 0.03 for plotting the ePSF difference data
            (``deltas=True``). If ``deltas=True``, the ``vmin`` is set
            to ``-vmax``. If ``deltas=False`` the ``vmin`` is set to
            ``vmax`` / 1e4.

        peak_norm : bool, optional
            Whether to normalize the ePSF data by the peak value. The
            default shows the ePSF flux per pixel.

        deltas : bool, optional
            Set to `True` to show the differences between each ePSF
            and the average ePSF.

        cmap : str or `matplotlib.colors.Colormap`, optional
            The colormap to use. The default is 'viridis'.

        dividers : bool, optional
            Whether to show divider lines between the ePSFs.

        divider_color, divider_ls : str, optional
            Matplotlib color and linestyle options for the divider
            lines between ePSFs. These keywords have no effect unless
            ``show_dividers=True``.

        figsize : (float, float), optional
            The figure (width, height) in inches.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The matplotlib figure object. This will be the current
            figure if ``ax=None``. Use ``fig.savefig()`` to save the
            figure to a file.

        Notes
        -----
        This method returns a figure object. If you are using this
        method in a script, you will need to call ``fig.show()`` to
        display the figure. If you are using this method in a Jupyter
        notebook, the figure will be displayed automatically.

        When in a notebook, if you do not store the return value of this
        function, the figure will be displayed twice due to the REPL
        automatically displaying the return value of the last function
        call. Alternatively, you can append a semicolon to the end of
        the function call to suppress the display of the return value.
        """
    return func


class _ModelGridPlotter:

    def __init__(self, model):
        self.model = model

    def _reshape_grid(self, data):
        pass

    @_plot_grid_docstring
    def plot_grid(self, *, ax=None, vmax_scale=None, peak_norm=False,
                  deltas=False, cmap='viridis', dividers=True,
                  divider_color='darkgray', divider_ls='-', figsize=None):
        pass
