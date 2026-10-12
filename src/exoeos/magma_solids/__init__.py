"""Native JAX dry mineral Gibbs models, in K, Pa, mol and J.

Each phase module provides COMPONENTS, standard_gibbs, mixing_gibbs and
gibbs. Ordered phases also expose order (site coordinates). Enable x64.
See documents/native_dry_solids.rst for bases, domains and provenance.
"""

from . import feldspar, oxides, pyroxene, spinel

__all__ = ["feldspar", "oxides", "pyroxene", "spinel"]
