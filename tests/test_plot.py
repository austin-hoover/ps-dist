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


def test_plot_rms_ellipse_cov():
    fig, ax = uplt.subplots()
    psv.plot_rms_ellipse_cov(cov_matrix=np.eye(2), level=[0.5, 1.0], ax=ax)
    ax.format(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    plt.savefig(os.path.join(output_dir, "fig_plot_rms_ellipse.png"))
    plt.close()


def test_plot_points_rms_ellipse():
    x = np.random.normal(size=(1000, 2))

    fig, ax = uplt.subplots()
    psv.plot(x, kind="scatter", color="black", ax=ax)
    psv.plot_rms_ellipse(x, level=[0.5, 1.0], color="red", ax=ax)
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


def test_plot_hist():
    x = np.random.normal(size=(10_000, 2))
    bin_edges = [np.linspace(-4.0, 4.0, 51), np.linspace(-4.0, 4.0, 51)]
    hist = ps.Histogram(edges=bin_edges)
    hist.bin(x)

    fig, ax = uplt.subplots(figwidth=3.0)
    psv.plot_hist(hist, ax=ax)

    plt.savefig(os.path.join(output_dir, "fig_plot_hist.png"))
    plt.close()


def test_plot_hist_prof():
    x = np.random.normal(size=(10_000, 2))
    bin_edges = [np.linspace(-4.0, 4.0, 51), np.linspace(-4.0, 4.0, 51)]
    hist = ps.Histogram(edges=bin_edges)
    hist.bin(x)

    fig, ax = uplt.subplots(figwidth=3.0)
    psv.plot_hist(hist, ax=ax, profx=True, profy=True)
    plt.savefig(os.path.join(output_dir, "fit_plot_hist_prof.png"))
    plt.close()


def test_plot_hist_rms_ellipse():
    x = np.random.normal(size=(10_000, 2))
    bin_edges = [np.linspace(-4.0, 4.0, 51), np.linspace(-4.0, 4.0, 51)]
    hist = ps.Histogram(edges=bin_edges)
    hist.bin(x)

    fig, ax = uplt.subplots(figwidth=3.0)
    psv.plot_hist(hist, ax=ax, rms_ellipse=True, rms_ellipse_kws=dict(level=[1.0, 2.0]))
    plt.savefig(os.path.join(output_dir, "fig_plot_hist_rms_ellipse.png"))
    plt.close()


def test_plot_hist():
    x = np.random.normal(size=(10_000, 2))
    bin_edges = [np.linspace(-4.0, 4.0, 51), np.linspace(-4.0, 4.0, 51)]

    fig, ax = uplt.subplots(figwidth=3.0)
    psv.plot(x, bins=bin_edges, kind="hist", ax=ax)
    plt.savefig(os.path.join(output_dir, "fig_plot_hist.png"))
    plt.close()


def test_plot_scatter():
    x = np.random.normal(size=(10_000, 2))

    fig, ax = uplt.subplots(figwidth=3.0)
    psv.plot(x, kind="scatter", ax=ax)
    plt.savefig(os.path.join(output_dir, "fig_plot_scatter.png"))
    plt.close()


def test_grid():
    pass
