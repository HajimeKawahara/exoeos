Build models and evidence
=========================

MCEG generates reStructuredText and CSS from the configuration in
``mceg/docs.yaml``. This repository owns the Sphinx build, theme and HTML output.
The generated pages need only Sphinx's standard Graphviz extension; they do not
import MCEG, MyST or scientific providers. The existing API documentation may
still import this package.

Install Graphviz (the ``dot`` executable), this package's documentation extras,
and a checkout of MCEG containing the ``docs`` command. With sibling checkouts
and Python 3.10 or newer:

.. code-block:: bash

   python -m pip install -e '.[docs]' -e '../mceg'
   ./update_doc.sh

The update script first generates RST, then builds the existing HTML docs.
To regenerate sources alone, without Sphinx or numerical execution:

.. code-block:: bash

   python -m mceg docs --config mceg/docs.yaml

Open ``documents/_build/html/mceg/index.html``. Generated sources and CSS live
in ``documents/mceg/_generated/`` and are ignored by Git. The generator's manifest
tracks its files, preserves neighboring authored files, and removes obsolete
owned pages. Edit the configuration or original records instead of generated RST.

Installed evidence catalog
--------------------------

The wheel includes ``exoeos/data/mceg/index.json`` and copies of the nine
registered contracts. MCEG versions with catalog support can inspect an installed
distribution without importing ExoEOS:

.. code-block:: bash

   mceg explain --package exoeos
   mceg graph --package exoeos

For a checkout or editable installation, replace ``--package exoeos`` with
``--catalog src/exoeos/data/mceg/index.json``. The catalog provides model-level
records and their explicit historical assessments, not species-specific coverage
or a check of a user's calculation. Repository evidence paths remain unchanged;
only bundled contract hashes are checked by package lookup.

After reviewing changes to the canonical contracts and catalog selections, run
``python mceg/build_catalog.py`` and commit the generated copies with the new
record version. This builder uses only the standard library. Original source
documents in local ``refs/`` are not included.

Inputs and interpretation
-------------------------

Paths in ``mceg/docs.yaml`` resolve relative to that configuration file.
``contracts`` selects provider roots and explicit contract patterns; ``cases``
selects authored calculation connections. ``source_root`` and ``output`` identify
the Sphinx tree and its dedicated generated directory. The catalog includes all
``mceg/contracts/*.yaml`` contracts in this repository. Archived reviews and
unregistered models are separate; the catalog is not a complete model inventory.

Each model has a short overview and separate Applicability, Assessments,
Evidence, References and Implementation & records pages. Evidence and Assessment
records have individual pages. Technical metadata can be expanded; unknown
support and source-review gaps remain visible. No current assessment is selected
from a historical list.

Local source downloads contain build-checkout files. Recorded implementation
revisions and artifact hashes are displayed without rechecking those bytes.
External citations are links, not newly inspected papers. Generation never
imports scientific implementations, executes solvers or changes assessments.

No calculation cases are selected by default. Add an authored case to the
configuration and include ``_generated/cases`` in the toctree when documenting
a particular atmosphere or interior calculation.
