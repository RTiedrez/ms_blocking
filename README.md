# ms_blocking

`ms_blocking` is a simple package built on `pandas`, `scipy`, and `networkx` that makes it easy to efficiently regroup rows of a pandas DataFrame based on equality of overlap between the elements of a column.
The tools it provides were designed with blocking (pre-regrouping rows in view of duplicate detection so you only save computing resources by comparing only rows inside the same 'block') in mind, but in spite of their simplicity, they could be enough for a full rule-based duplicate detection or clustering pipeline.
## Installation

```bash
$ pip install ms_blocking
```

## Usage

```python
import ms_blocking.ms_blocking as msb
from ms_blocking.datasets import get_users

df = get_users()

city_blocker = msb.AttributeEquivalenceBlocker(["City"])
links = city_blocker.block(df)
msb.add_blocks_to_dataset(df, links)
```

## License

`ms_blocking` was created by RTiedrez. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`ms_blocking` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

Our work is mostly based on [`py_entitymatching`](https://github.com/anhaidgroup/py_entitymatching)'s blocking features. Our aim was to create a fast, lightweight, portable, and RAM-preserving interface to said features, which we accomplished by redesigning `ms_blocking` as a wrapper around `pandas`, `scipy`, and `networkx` methods.