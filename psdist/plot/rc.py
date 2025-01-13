import matplotlib
import matplotlib.pyplot as plt

BLACK = "black"
CYCLE = "colorblind"
CMAPCYC = "twilight"
CMAPDIV = "BuRd"
CMAPSEQ = "Viridis"
CMAPCAT = "colorblind10"
DIVERGING = "div"
FRAMEALPHA = 0.8  # legend and colorbar
FONTNAME = "sans-serif"
FONTSIZE = 9.0
GRIDALPHA = 0.0
GRIDBELOW = "line"
GRIDPAD = 3.0
GRIDRATIO = 0.5  # differentiated from major by half size reduction
GRIDSTYLE = "-"
LABELPAD = 4.0  # default is 4.0, previously was 3.0
LARGESIZE = "med-large"
LINEWIDTH = 0.6
MARGIN = 0.05
MATHTEXT = False
SMALLSIZE = "medium"
TICKDIR = "out"
TICKLEN = 4.0
TICKLENRATIO = 0.5  # very noticeable length reduction
TICKMINOR = True
TICKPAD = 2.0
TICKWIDTHRATIO = 0.8  # very slight width reduction
TITLEPAD = 5.0  # default is 6.0, previously was 3.0
WHITE = "white"
ZLINES = 2  # default zorder for lines
ZPATCHES = 1

_rc_matplotlib_default = {
    "axes.axisbelow": GRIDBELOW,
    "axes.formatter.use_mathtext": MATHTEXT,
    "axes.grid": False,  # enable lightweight transparent grid by default
    "axes.grid.which": "major",
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "axes.labelpad": LABELPAD,  # more compact
    "axes.labelsize": SMALLSIZE,
    "axes.labelweight": "normal",
    "axes.linewidth": LINEWIDTH,
    "axes.titlepad": TITLEPAD,  # more compact
    "axes.titlesize": LARGESIZE,
    "axes.titleweight": "normal",
    "axes.xmargin": MARGIN,
    "axes.ymargin": MARGIN,
    "errorbar.capsize": 3.0,
    "figure.autolayout": False,
    "figure.figsize": (4.0, 4.0),  # for interactife backends
    "figure.dpi": 100,
    "figure.facecolor": "white",  # similar to MATLAB interface
    "figure.titlesize": LARGESIZE,
    "figure.titleweight": "bold",  # differentiate from axes titles
    "font.serif": [
        "TeX Gyre Schola",  # Century lookalike
        "TeX Gyre Bonum",  # Bookman lookalike
        "TeX Gyre Termes",  # Times New Roman lookalike
        "TeX Gyre Pagella",  # Palatino lookalike
        "DejaVu Serif",
        "Bitstream Vera Serif",
        "Computer Modern Roman",
        "Bookman",
        "Century Schoolbook L",
        "Charter",
        "ITC Bookman",
        "New Century Schoolbook",
        "Nimbus Roman No9 L",
        "Noto Serif",
        "Palatino",
        "Source Serif Pro",
        "Times New Roman",
        "Times",
        "Utopia",
        "serif",
    ],
    "font.sans-serif": [
        "TeX Gyre Heros",  # Helvetica lookalike
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Arial",
        "Avenir",
        "Fira Math",
        "Fira Sans",
        "Frutiger",
        "Geneva",
        "Gill Sans",
        "Helvetica",
        "Lucid",
        "Lucida Grande",
        "Myriad Pro",
        "Noto Sans",
        "Roboto",
        "Source Sans Pro",
        "Tahoma",
        "Trebuchet MS",
        "Ubuntu",
        "Univers",
        "Verdana",
        "sans-serif",
    ],
    "font.cursive": [
        "TeX Gyre Chorus",  # Chancery lookalike
        "Apple Chancery",
        "Felipa",
        "Sand",
        "Script MT",
        "Textile",
        "Zapf Chancery",
        "cursive",
    ],
    "font.fantasy": [
        "TeX Gyre Adventor",  # Avant Garde lookalike
        "Avant Garde",
        "Charcoal",
        "Chicago",
        "Comic Sans MS",
        "Futura",
        "Humor Sans",
        "Impact",
        "Optima",
        "Western",
        "xkcd",
        "fantasy",
    ],
    "font.monospace": [
        "TeX Gyre Cursor",  # Courier lookalike
        "DejaVu Sans Mono",
        "Bitstream Vera Sans Mono",
        "Computer Modern Typewriter",
        "Andale Mono",
        "Courier New",
        "Courier",
        "Fixed",
        "Nimbus Mono L",
        "Terminal",
        "monospace",
    ],
    "font.family": FONTNAME,
    "font.size": FONTSIZE,
    "grid.alpha": GRIDALPHA,  # lightweight unobtrusive gridlines
    "grid.color": BLACK,  # lightweight unobtrusive gridlines
    "grid.linestyle": GRIDSTYLE,
    "grid.linewidth": LINEWIDTH,
    "hatch.color": BLACK,
    "hatch.linewidth": LINEWIDTH,
    "image.cmap": CMAPSEQ,
    "image.interpolation": "none",
    "lines.linestyle": "-",
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "legend.borderaxespad": 0,  # i.e. flush against edge
    "legend.borderpad": 0.5,  # a bit more roomy
    "legend.columnspacing": 1.5,  # a bit more compact (see handletextpad)
    "legend.edgecolor": BLACK,
    "legend.facecolor": WHITE,
    "legend.fancybox": False,  # i.e. BboxStyle 'square' not 'round'
    "legend.fontsize": SMALLSIZE,
    "legend.framealpha": FRAMEALPHA,
    "legend.handleheight": 1.0,  # default is 0.7
    "legend.handlelength": 2.0,  # default is 2.0
    "legend.handletextpad": 0.5,  # a bit more compact (see columnspacing)
    "mathtext.default": "it",
    "mathtext.fontset": "custom",
    "mathtext.bf": "regular:bold",  # custom settings implemented above
    "mathtext.cal": "cursive",
    "mathtext.it": "regular:italic",
    "mathtext.rm": "regular",
    "mathtext.sf": "regular",
    "mathtext.tt": "monospace",
    "patch.linewidth": LINEWIDTH,
    "savefig.bbox": None,  # do not use 'tight'
    "savefig.directory": "",  # use the working directory
    "savefig.dpi": 1000,  # use academic journal recommendation
    "savefig.facecolor": WHITE,  # use white instead of 'auto'
    "savefig.format": "pdf",  # use vector graphics
    "savefig.transparent": False,
    "xtick.color": BLACK,
    "xtick.direction": TICKDIR,
    "xtick.labelsize": SMALLSIZE,
    "xtick.major.pad": TICKPAD,
    "xtick.major.size": TICKLEN,
    "xtick.major.width": LINEWIDTH,
    "xtick.minor.pad": TICKPAD,
    "xtick.minor.size": TICKLEN * TICKLENRATIO,
    "xtick.minor.width": LINEWIDTH * TICKWIDTHRATIO,
    "xtick.minor.visible": TICKMINOR,
    "ytick.color": BLACK,
    "ytick.direction": TICKDIR,
    "ytick.labelsize": SMALLSIZE,
    "ytick.major.pad": TICKPAD,
    "ytick.major.size": TICKLEN,
    "ytick.major.width": LINEWIDTH,
    "ytick.minor.pad": TICKPAD,
    "ytick.minor.size": TICKLEN * TICKLENRATIO,
    "ytick.minor.width": LINEWIDTH * TICKWIDTHRATIO,
    "ytick.minor.visible": TICKMINOR,
}


def set_rcparams():
    for key, val in _rc_matplotlib_default.items():
        plt.rcParams[key] = val
