# Welcome to gingado!


`gingado` seeks to facilitate the use of machine learning in economic
and finance use cases, while promoting good practices. This package aims
to be suitable for beginners and advanced users alike. Use cases may
range from simple data retrievals to experimentation with machine
learning algorithms to more complex model pipelines used in production.

## Overview

`gingado` is a free, open source library built different
functionalities:

- [**data
  augmentation**](https://bis-med-it.github.io/gingado/augmentation.html),
  to add data from official sources, improving the machine models being
  trained by the user;

- **relevant**
  [**datasets**](https://bis-med-it.github.io/gingado/datasets.html),
  both real and simulated, to allow for easier model development and
  comparison;

- **automatic** [**benchmark
  model**](https://bis-med-it.github.io/gingado/benchmark.html), to
  assess candidate models against a reasonably well-performant model;

- **machine learning-based**
  [**estimators**](https://bis-med-it.github.io/gingado/estimators.html),
  to help answer questions of academic or practical importance;

- **support for** [**model
  documentation**](https://bis-med-it.github.io/gingado/documentation.html),
  to embed documentation and ethical considerations in the model
  development phase; and

- [**utilities**](https://bis-med-it.github.io/gingado/utils.html),
  including tools to allow for lagging variables in a straightforward
  way.

Each of these functionalities builds on top of the previous one. They
can be used on a stand-alone basis, together, or even as part of a
larger pipeline from data input to model training to documentation!

> [!TIP]
>
> New functionalities are planned over time, so consider checking
> frequently on `gingado` for the latest toolsets.

## Install

> [!NOTE]
>
> Please make sure you have read and understood the license disclaimer
> in the NOTES.md file in our [GitHub
> repository](https://github.com/bis-med-it/gingado) before using
> gingado.

To install `gingado`, simply run the following code on the terminal:

`$ pip install gingado`

## Attribution

If you use this package in your work, please consider citing Araujo
(2023).

In BibTeX format:

    @techreport{gingado,
        author = {Araujo, Douglas KG},
        title = {gingado: a machine learning library focused on economics and finance},
        series = {BIS Working Paper},
        type = {Working Paper},
        institution = {Bank for International Settlements},
        year = {2023},
        number = {1122}
    }

Over time, new tools that are described in specific papers might be
added (eg, a machine learning-based econometric estimator). Please
consider citing them as well if used in your work. Specific information,
if any, can be found in the documentation.

## Design principles

The choices made during development of `gingado` derive from the
following principles, in no particular order:

- **flexibility**: users can use `gingado` out of the box or build
  custom processes on top of it;

- **compatibility**: `gingado` works well with other widely used
  libraries in machine learning, such as `scikit-learn` and `pandas`;
  and

- **responsibility**: `gingado` facilitates and promotes model
  documentation, including ethical considerations, as part of the
  machine learning development workflow.

For more information about `gingado`, please read the
[paper](https://www.bis.org/publ/work1122.pdf).

## Acknowledgements

`gingado`’s API is inspired on the following libraries:

- `scikit-learn` (Buitinck et al. 2013)

- `keras` (website [here](https://keras.io/about/) and also, [this
  essay](https://medium.com/s/story/notes-to-myself-on-software-engineering-c890f16f4e4d))

- `fastai` (Howard and Gugger 2020)

In addition, `gingado` is developed and maintained using
[`quarto`](https://quarto.org/).

## References

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-gingado" class="csl-entry">

Araujo, Douglas KG. 2023. “Gingado: A Machine Learning Library Focused
on Economics and Finance.” Working Paper 1122. BIS Working Paper. Bank
for International Settlements.

</div>

<div id="ref-sklearnAPI" class="csl-entry">

Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa,
Andreas Mueller, Olivier Grisel, Vlad Niculae, et al. 2013. “API Design
for Machine Learning Software: Experiences from the Scikit-Learn
Project.” *CoRR* abs/1309.0238. <http://arxiv.org/abs/1309.0238>.

</div>

<div id="ref-fastaiAPI" class="csl-entry">

Howard, Jeremy, and Sylvain Gugger. 2020. “Fastai: A Layered API for
Deep Learning.” *Information* 11 (2).
<https://doi.org/10.3390/info11020108>.

</div>

</div>
