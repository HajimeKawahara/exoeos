"""Native JAX dry and hydrous mineral Gibbs models, in K, Pa, mol and J.

Each phase module provides COMPONENTS, standard_gibbs, mixing_gibbs and
gibbs. Ordered phases also expose order (site coordinates). Enable x64.
See documents/native_{dry,hydrous}_solids.rst for domains and provenance.
"""

from . import amphibole, biotite, feldspar, hornblende, oxides, pyroxene, spinel

__all__ = ["amphibole", "biotite", "feldspar", "hornblende", "oxides", "pyroxene", "spinel"]
