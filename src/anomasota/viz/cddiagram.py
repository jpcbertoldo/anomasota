from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.patches as mpl_patches
import networkx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class CDDiagramDisplay:
    """Critical Difference Diagram"""
    # TODO generate latex code for tikzpicture
    # alternatives to look at
    # matplotlib2tikz DEPRECATED, became tikzplotlib
    # tikzplotlib
    
    tmp_rcparams: ClassVar[Dict[str, str]] = {
        "mathtext.fontset": "custom",
        "mathtext.rm": "Bitstream Vera Sans",
        "mathtext.it": "Bitstream Vera Sans:italic",
        "mathtext.bf": "Bitstream Vera Sans:bold",
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }   
    
    average_ranks: Union[np.ndarray, List[float]] = field(repr=False)
    methods_names: Union[np.ndarray, List[str]] = field(repr=False)
    cliques: List[Tuple[int]] = field(repr=False)
    pvalues_matrix: np.ndarray = field(repr=False)
    alphas_matrix: np.ndarray = field(repr=False)
    metric_name: str = None
    
    # attributes from __post_init__()
    num_methods: int = field(init=False)
    num_cliques: int = field(init=False)
    argsort_: np.ndarray = field(init=False, repr=False)
    
    # attributes from plot()
    ax_: mpl.axes.Axes = field(init=False, repr=False)
    fig_: mpl.figure.Figure = field(init=False, repr=False)
    axlegend_: mpl.axes.Axes = field(init=False, repr=False)
    figlegend_: mpl.figure.Figure = field(init=False, repr=False)
    title_: str = field(init=False, repr=False)
    simplified_names_mapping_: Dict[str, str] = field(init=False, repr=False)
    simplified_names_inv_mapping_: Dict[str, str] = field(init=False, repr=False)
    
    def __post_init__(self):
        
        # TODO validate inputs
        
        # TODO rename computed attributes to have _ suffix
        self.num_methods = len(self.methods_names)
        self.num_cliques = len(self.cliques)
        
        self.average_ranks = np.asarray(self.average_ranks)
        self.methods_names = np.asarray(self.methods_names)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING
        # this sorting is very important
        # if the argsort is not propertly applied to the cliques/matrices the diagram will be wrong  

        self.argsort_ = argsort = np.argsort(self.average_ranks)
        self.average_ranks = self.average_ranks.copy()[argsort]
        self.methods_names = self.methods_names.copy()[argsort]

        def modify_idx(unsorted_idx: int) -> int:
            """Replace an index in the unsorted array with the index in the sorted array."""
            return np.where(argsort == unsorted_idx)[0][0]

        def modify_clique(unsorted_idx_clique: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(modify_idx(idx) for idx in unsorted_idx_clique)

        self.cliques = sorted(modify_clique(cliq) for cliq in self.cliques)
        
        def modify_matrix(unsorted_matrix: np.ndarray) -> np.ndarray:
            res = np.zeros_like(unsorted_matrix)
            for i, j in np.ndindex(unsorted_matrix.shape):
                res[i, j] = unsorted_matrix[argsort[i], argsort[j]]
            return res
        
        self.pvalues_matrix = modify_matrix(self.pvalues_matrix)
        self.alphas_matrix = modify_matrix(self.alphas_matrix)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SORTING
        
        self._validate()
    
    def _validate(self):
        pass  # TODO

    def plot(
        self, 
        ax, axlegend=None, 
        title: str = None,
        avgrank_formatter: Optional[Callable[[float], str]] = None,
        pvalue_formatter: Optional[Callable[[float], str]] = None,
        alpha_formatter: Optional[Callable[[float], str]] = None,
        simplify_names: Optional[Union[bool, Dict[str, str], Callable[[str], str]]] = False,
        stem_kwargs: Dict[str, Any]={}, 
        clique_kwargs: Dict[str, Any]={}, 
        pvalue_kwargs: Dict[str, Any]={}, 
        alpha_kwargs: Dict[str, Any] = {}, 
        rcparams: Dict[str, Any] = {}, 
        **kwargs,
    ):
        """Plot visualization
        
        TODO: make pvalues optional
        TODO: make (corrected, pairwise) alphas optional
        TODO: make annotation of wilcoxon alternative
        TODO: make annotation of alpha
        
        simplify_names: bool, dict, callable, default=False
            False: do not simplify
            None or True: use default simplifier, i.e. names are replaced by A, B, C, etc 
                (see `default_name_simplifier_factory`)
            Dict: use the dict as a mapping, keys are original names, values are simplified names
            Callable: use the callable to simplify, it should take a string and return a string
        
        Returns
        -------
        self
        """
        
        self.ax_ = ax
        self.fig_ = ax.figure
        self.axlegend_ = axlegend
        self.figlegend_ = axlegend.figure if axlegend is not None else None
        
        self.title_ = title if title is not None else f"Critical Difference Diagram of '{self.metric_name}'" if self.metric_name is not None else "Critical Difference Diagram"
        
        avgrank_formatter = avgrank_formatter or self.default_avgrank_formatter
        pvalue_formatter = pvalue_formatter or self.default_pvalue_formatter
        alpha_formatter = alpha_formatter or self.default_alpha_formatter

        if simplify_names is False:
            name_simplifier = None
            
        elif simplify_names is None or simplify_names is True:
            name_simplifier = self.default_name_simplifier_factory(self.methods_names)  # default simplifier
        
        elif isinstance(simplify_names, dict):
            name_simplifier = lambda name: name_simplifier.get(name) or name  # dict lookup
        
        elif iscallable(simplify_names):
            name_simplifier = simplify_names
            
        else:
            raise ValueError(f"Invalid type for ``simplify_names``: {type(simplify_names)}")
        
        if name_simplifier is not None:
            simplified_names = [name_simplifier(name) for name in self.methods_names]
            self.simplified_names_inv_mapping_ = dict(zip(self.methods_names, simplified_names))
            self.simplified_names_mapping_ = dict(zip(simplified_names, self.methods_names))
            simplified_names = np.array(simplified_names)
            legend_rows = [
                f"{simplename}: {self.simplified_names_mapping_[simplename]}" 
                for simplename in simplified_names
            ]
            
        else: 
            simplified_names = self.methods_names.copy() 
            self.simplified_names_inv_mapping_ = self.simplified_names_mapping_ = legend_rows = None
                
        rcparams = {**{
            "mathtext.fontset": "custom",
            "mathtext.rm": "Bitstream Vera Sans",
            "mathtext.it": "Bitstream Vera Sans:italic",
            "mathtext.bf": "Bitstream Vera Sans:bold",
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }, **rcparams}
        
        CDDiagramDisplay._plot(
            ax,
            self.title_, 
            self.average_ranks, simplified_names, self.cliques, self.pvalues_matrix, self.alphas_matrix,
            avgrank_formatter, pvalue_formatter, alpha_formatter,
            stem_kwargs, clique_kwargs, pvalue_kwargs, alpha_kwargs, 
            rcparams, 
            **kwargs,
        )
        
        if axlegend is not None and legend_rows is not None:
            CDDiagramDisplay._plot_legend(axlegend, legend_rows, rcparams)
        
        return self
    
    @staticmethod
    def _plot(
        ax, 
        title, 
        average_ranks, methods_names, cliques, pvalues_matrix, alphas_matrix,
        avgrank_formatter, pvalue_formatter, alpha_formatter,
        stem_kwargs, clique_kwargs, pvalue_kwargs, alpha_kwargs, 
        rcparams, **kwargs,
    ):
        # ------------------------------ CONFIGS ------------------------------
        num_methods = len(methods_names)
        num_cliques = len(cliques)
        
        assert average_ranks.shape == (num_methods,), f"{average_ranks.shape} != ({num_methods},)"
        assert pvalues_matrix.shape == (num_methods, num_methods), f"{pvalues_matrix.shape} != ({num_methods}, {num_methods})"
        assert alphas_matrix.shape == (num_methods, num_methods), f"{alphas_matrix.shape} != ({num_methods}, {num_methods})"
        assert all(1 < len(clique) <= num_methods for clique in cliques), f"Invalid cliques: {cliques}"
        assert all(0 <= idx < num_methods for clique in cliques for idx in clique), f"Invalid cliques: {cliques}"
        assert ((average_ranks >= 1) & (average_ranks <= num_methods)).all(), f"Invalid average ranks: {average_ranks}"
        assert all(methods_names), f"Invalid methods names: {methods_names}"
        assert callable(avgrank_formatter), f"{type(avgrank_formatter)}"
        assert callable(pvalue_formatter), f"{type(pvalue_formatter)}"
        assert callable(alpha_formatter), f"{type(alpha_formatter)}"
        
        # ------------------------------ KWARGS ------------------------------
        STEM_VSPACE_BETWEEN = stem_kwargs.pop("STEM_VSPACE_BETWEEN", 0.12)
        STEM_TEXT_XYSPACE_CONST = stem_kwargs.pop("STEM_TEXT_XYSPACE_CONST", 4.0)
        
        CLIQUE_VMARGIN_BOTTOM = clique_kwargs.pop("CLIQUE_VMARGIN_BOTTOM", 0.05)
        CLIQUE_VMARGIN_TOP = clique_kwargs.pop("CLIQUE_VMARGIN_TOP", 0.3)
        CLIQUE_VSPACE_BETWEEN = clique_kwargs.pop("CLIQUE_VSPACE_BETWEEN", 0.08)

        PVALUES_ARROW_HEAD_SIZE = stem_kwargs.pop("PVALUES_ARROW_HEAD_SIZE", 0.03)
        PVALUES_HEIGHT_SHIFT = stem_kwargs.pop("PVALUES_HEIGHT_SHIFT", 0.02)

        TITLE_PAD = kwargs.pop("TITLE_PAD", 25)
        XTICKS_PAD = kwargs.pop("XTICKS_PAD", -15)
        
        # ------------------------ COMPUTE PARAMETERS ------------------------
        
        # cliques are linearly spaced 
        # from CLIQUE_VMARGIN_BOTTOM 
        #   to CLIQUE_VMARGIN_BOTTOM + CLIQUE_VSPACE_BETWEEN * (num_cliques - 1) = 'last-clique-height'
        clique_heights = CLIQUE_VMARGIN_BOTTOM + CLIQUE_VSPACE_BETWEEN * np.arange(0, num_cliques)
        last_clique_height = clique_heights[-1] if num_cliques > 0 else 0.0
        
        # stems are linearly spaced until the stem in the middle of the diagram
        # then they are mirrored in the middle, going back to the first stem height
        # so the heights range is +/-
        # from 'last-clique-height' + CLIQUE_VMARGIN_TOP = 'first-stem-height'
        #   to 'first-stem-height' + STEM_VSPACE_BETWEEN * (num_methods - 1) // 2
        STEM_MIN_HEIGHT = last_clique_height + CLIQUE_VMARGIN_TOP
        stem_heights = STEM_MIN_HEIGHT + STEM_VSPACE_BETWEEN * np.arange(0, num_methods)

        # mirror the stem heights in the middle
        if num_methods % 2 == 0:
            idx_right = idx_left = num_methods // 2
        else:
            idx_right = int(np.floor(num_methods / 2))
            idx_left = idx_right + 1
            
        stem_heights[idx_left:] = np.flip(stem_heights[:idx_right])
        
        # TODO CHANGE THIS NAME TO NOMINAL HEIGHT
        # pvalues are between the stems and the clique, nominally in the middle
        # ('last-clique-height' + 'first-stem-height') / 2 = 'pvalues-nominal-height'
        # and to avoid alignment, actual heights are shifted + or - PVALUES_HEIGHT_SHIFT
        # 'pvalues-height' = 'pvalues-nominal-height' +- PVALUES_HEIGHT_SHIFT
        PVALUES_HEIGHT_NOMINAL = (last_clique_height + stem_heights[0]) / 2
        
        # -------------------------------- PLOT --------------------------------
        with mpl.rc_context(rc=rcparams):
            # stems (average ranks)
            stem_kwargs = {**dict(linewidth=1, color="black"), **stem_kwargs}
            ax.vlines(average_ranks, 0, stem_heights, **stem_kwargs)

            # stems' annotations (method name and average rank beside it)
            for idx in range(num_methods):
                name = methods_names[idx]
                rank = average_ranks[idx]
                rankstr = avgrank_formatter(rank)
                height = stem_heights[idx]
                xy = (rank, height)
                halign = "left" if idx < idx_left else "right"
                xtext_sign = 1 if idx < idx_left else -1
                xytext = (xtext_sign * STEM_TEXT_XYSPACE_CONST, -STEM_TEXT_XYSPACE_CONST)
                
                # method name
                ax.annotate(name, xy=xy, xytext=xytext, textcoords="offset points", va="bottom", ha=halign, color="black", fontsize="medium")
                
                # invert the horizontal alignment and text position
                xytext = (-1 * xytext[0], xytext[1])
                halign = "right" if halign == "left" else "left"
                
                # average rank
                ax.annotate(rankstr, xy=xy, xytext=xytext, textcoords="offset points", va="bottom", ha=halign, color="red", fontsize="small")
                
                # annotate first and last because that is where the scale gets distorted
                if not (idx == 0 or idx == num_methods - 1):
                    continue

                xy = (rank, 0)
                xytext = (0, STEM_TEXT_XYSPACE_CONST)
                ax.annotate(rankstr, xy=xy, xytext=xytext, textcoords="offset points", va="bottom", ha="center", color="gray", fontsize="x-small")
                
            # cliques (groups of methods without pairwise-ly significant differences)
            for cliqidx, cliq in enumerate(cliques):
                minrank = average_ranks[min(cliq)]
                maxrank = average_ranks[max(cliq)]
                clique_height = clique_heights[cliqidx]
                ax.hlines(clique_height, minrank, maxrank, linewidth=2, color="black")
                for idx in cliq:
                    rank = average_ranks[idx]
                    ax.plot(rank, clique_height, marker="o", color="black", markerfacecolor="white", markersize=4)

            # pvalues (significance of pairwise differences) between adjacent stems
            for idx in range(num_methods - 1):
                rank = average_ranks[idx]
                idx_next = idx + 1
                rank_next = average_ranks[idx_next]
                shift_sign = 1 if idx % 2 == 0 else -1
                height = PVALUES_HEIGHT_NOMINAL + shift_sign * PVALUES_HEIGHT_SHIFT

                # arrows
                # one cannot make a double-headed arrow with ax.arrows, so we have to do it manually
                def double_headed_arrow(ax_, x, y, dx, dy, **kwargs):
                    ax_.arrow(x, y, dx, dy, **kwargs)
                    ax_.arrow(x + dx, y + dy, -dx, -dy, **kwargs)
                
                double_headed_arrow(
                    ax, rank, height, rank_next - rank, 0, 
                    head_width=PVALUES_ARROW_HEAD_SIZE, 
                    head_length=PVALUES_ARROW_HEAD_SIZE, 
                    length_includes_head=True, 
                    fc='gray', ec='gray', linewidth=0.5, shape="full"
                )
                
                # pvalue and alpha annotations
                xy = ((rank + rank_next) / 2, height)
                pvaluestr = pvalue_formatter(pvalues_matrix[idx, idx_next])
                alphastr = alpha_formatter(alphas_matrix[idx, idx_next])
                ax.annotate(pvaluestr, xy=xy, xytext=(0, 0), textcoords="offset points", va="bottom", ha="center", color="gray", fontsize="small")
                ax.annotate(alphastr, xy=xy, xytext=(0, -1), textcoords="offset points", va="top", ha="center", color="gray", fontsize="xx-small")
                
            # axes
            ax.set_title(title, pad=TITLE_PAD)
            
            # modify x-axis limits to show all stems and spare some space on the left and right by ditoring the scale
            ax.set_xlim(1, num_methods)
            def fwd(x):
                clipped = np.clip(x, min(average_ranks), max(average_ranks))
                return clipped + (x - clipped) * 0.1
            # inverse is identity, i dont know why it works :)
            ax.set_xscale('function', functions=(fwd, lambda y: y ))
            ax.set_yscale('linear')
            ax.set_aspect("equal")
            
            # hide y-axis and splines except bottom one
            ax.yaxis.set_visible(False)
            ax.spines[["left", "top", "right"]].set_visible(False)
            ax.spines["bottom"].set_position("zero")
            
            # set x-axis ticks
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.tick_params(axis="x", direction="in", pad=XTICKS_PAD)
                        
            # invert both axes
            ax.xaxis.set_inverted(True)
            ax.yaxis.set_inverted(True)
            
    @staticmethod
    def _plot_legend(axlegend, legend_rows, rcparams):
        with mpl.rc_context(rc=rcparams):
            handles = [
                Line2D([0], [0], color="black", lw=0, label=row) 
                for row in legend_rows
            ]
            axlegend.legend(handles=handles, fontsize="small", title="Methods", title_fontsize="small", frameon=False, loc="center")
            axlegend.axis("off")
        
    @staticmethod
    def default_avgrank_formatter(x: float) -> str:
        return f"{x:.2f}"

    @staticmethod
    def default_pvalue_formatter(x: float) -> str:
        significant_digits = float(f"{x:.3g}")
        return f"$p={significant_digits:0.2%}$%"

    @staticmethod
    def default_alpha_formatter(x: float) -> str:
        significant_digits = float(f"{x:.3g}")
        return fr"$\alpha={significant_digits:0.2%}$%"

    @staticmethod
    def default_name_simplifier_factory(original_names: List[str]) -> Callable[[str], str]:
        
        ALPHABET = list("abcdefghijklmnopqrstuvwxyz".upper())
        # extend it just in case
        ALPHABET = ALPHABET + ["".join(x) for x in product(ALPHABET, ALPHABET)]
        mapping = dict(zip(original_names, ALPHABET[:len(original_names)]))
        
        def default_name_simplifier(name: str) -> str:
            return mapping[name]
            
        return default_name_simplifier

# TODO make a 2D version like https://github.com/mirkobunse/CriticalDifferenceDiagrams.jl/blob/main/docs/src/assets/2d_example.svg
# how can i represent the cliques AND pvalues?
# this example image doesnt have overlapping cliques, so it's easy...
