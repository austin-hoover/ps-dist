import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import ultraplot as uplt

import psdist as ps
import psdist.plot as psv


uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["grid"] = False
uplt.rc["savefig.dpi"] = 200


path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


def test_plot_ellipse():
    fig, ax = uplt.subplots()
    psv.plot_ellipse(r1=1.5, r2=0.5, ax=ax)
    ax.format(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    plt.savefig(os.path.join(output_dir, "fig_plot_ellipse.png"))
    plt.close()


def test_plot_circle():
    fig, ax = uplt.subplots()
    psv.plot_circle(ax=ax)
    ax.format(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    plt.savefig(os.path.join(output_dir, "fig_plot_circle.png"))


def test_plot_rms_ellipse():
    fig, ax = uplt.subplots()
    psv.plot_rms_ellipse(cov_matrix=np.eye(2), level=[0.5, 1.0], ax=ax)
    ax.format(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    plt.savefig(os.path.join(output_dir, "fig_plot_rms_ellipse.png"))
    plt.close()


def test_plot_hist_1d():
    x = np.random.normal(size=10_000)
    bin_edges = np.linspace(-4.0, 4.0, 51)
    hist = ps.Histogram1D(edges=bin_edges)
    hist.bin(x)

    fig, ax = uplt.subplots(figsize=(3.0, 1.5))
    psv.plot_hist_1d(hist, kind="step", ax=ax, color="black")

    plt.savefig(os.path.join(output_dir, "fig_plot_profile.png"))
    plt.close()


def test_plot_hist_overlay():
    x = np.random.normal(size=(10_000, 2))
    bin_edges = [
        np.linspace(-4.0, 4.0, 51),
        np.linspace(-4.0, 4.0, 51)
    ]
    hist = ps.Histogram(edges=bin_edges)
    hist.bin(x)

    fig, ax = uplt.subplots(figwidth=3.0)
    ax.pcolormesh(hist.coords[0], hist.coords[1], hist.values.T)

    psv.hist.plot_profiles_overlay(hist, profx=True, profy=True, ax=ax)

    plt.savefig(os.path.join(output_dir, "fig_plot_hist_overlay.png"))
    plt.close()

