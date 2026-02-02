# ms_blocking

`ms_blocking` is a simple package built on `pandas`, `scipy`, and `networkx` that makes it easy to efficiently regroup rows of a pandas DataFrame based on equality of overlap between the elements of a column.

'Blocking' refers to the (computationaly cheap) pre-regrouping of rows in a dataset in view of duplicate detection. Once rows are regrouped inside "blocks", more expensive duplicates detection algorithms can be run, but since only rows that belong to the same "block" are compared, both RAM and execution time are saved.


Even though `ms_blocking` was designed with blocking in mind, it may serve as a standalone, lightweight, and fast rule-based duplicates detection toolkit.
## Installation

```bash
$ pip install ms_blocking
```

## Usage

```python
import ms_blocking.ms_blocking as msb
from ms_blocking.datasets import get_users

df = get_users()
df
```
| id | Name                | City              | Age | websites                                                                           |
|----|---------------------|-------------------|-----|------------------------------------------------------------------------------------|
| 0  | Jean d'Aux          | Lille             | 26  | ['jeandaux.fr', 'lillefans.fr']                                                    |
| 1  | Jacques Dupond      | Villeneuve d'Ascq | 37  | ['somewebsite.com/users/jacquesdupond', 'jacquesdupond.fr', 'pythonensamusant.fr'] |
| 2  | Pierre Dusquesnes   | Phalempin         | 24  | ['somewebsite.com/users/rpz59']                                                    |
| 3  | Paul Delarue        | Roubaix           | 32  | ['roubaixlove.fr']                                                                 |
| 4  | Jacques Dupont      | Villeneuve d'Ascq | 37  | ['jacquesdupond.fr']                                                               |
| 5  | pierre dusquesnes   | Phalempin         | 24  | []                                                                                 |
| 6  | Jean-Michel Python  | Douai             | 49  | ['lensfans.fr', 'pythonensamusant.fr']                                             |
| 7  | Gédéon Glincarné    | Paris             | 53  | ['lorem.fr']                                                                       |
| 8  | Sophie Delarue      | Roubaix           | 33  | []                                                                                 |
| 9  | Jeanne Verbrugge    | Valenciennes      | 41  | ['somewebsite.com/users/jajanne59']                                                |
| 10 | Caroline Dufour     | Lens              | 45  | ['pythonensamusant.fr', 'lensfans.fr']                                             |
| 11 | sophie_delarue      | Roubaix           | 33  | []                                                                                 |
| 12 | Marcel Vandermersch | Fourmies          | 48  | ['lesrecettesdemarcel.fr']                                                         |
| 13 | Benoît Benoît       | Lens              | 15  | ['lensfans.fr']                                                                    |

```
city_blocker = msb.AttributeEquivalenceBlocker(["City"])
age_blocker = msb.AttributeEquivalenceBlocker(["Age"])
websites_blocker = msb.OverlapBlocker(["websites"])
final_blocker = (city_blocker & age_blocker) | websites_blocker

links = final_blocker.block(df, motives=True)

report = msb.add_blocks_to_dataset(
    df,
    links,
    motives=True,
    show_as_pairs=True,
    output_columns=["id", "Name"],
    merge_blocks=False,
)
report
```
| id_l | Name_l            | id_r | Name_r             | _block | _motive                                                    |
|------|-------------------|------|--------------------|--------|------------------------------------------------------------|
| 1    | Jacques Dupond    | 4    | Jacques Dupont     | 0      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 1    | Jacques Dupond    | 6    | Jean-Michel Python | 0      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 1    | Jacques Dupond    | 10   | Caroline Dufour    | 0      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 1    | Jacques Dupond    | 4    | Jacques Dupont     | 1      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 1    | Jacques Dupond    | 6    | Jean-Michel Python | 1      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 1    | Jacques Dupond    | 10   | Caroline Dufour    | 1      | {"Same 'Age'", ">=1 overlap in 'websites'", "Same 'City'"} |
| 10   | Caroline Dufour   | 6    | Jean-Michel Python | 1      | {">=1 overlap in 'websites'"}                              |
| 10   | Caroline Dufour   | 13   | Benoît Benoît      | 1      | {">=1 overlap in 'websites'"}                              |
| 2    | Pierre Dusquesnes | 5    | pierre dusquesnes  | 2      | {"Same 'Age'", "Same 'City'"}                              |
| 8    | Sophie Delarue    | 11   | sophie_delarue     | 3      | {"Same 'Age'", "Same 'City'"}                              |
| 10   | Caroline Dufour   | 6    | Jean-Michel Python | 4      | {">=1 overlap in 'websites'"}                              |
| 10   | Caroline Dufour   | 13   | Benoît Benoît      | 4      | {">=1 overlap in 'websites'"}                              |
| 13   | Benoît Benoît     | 6    | Jean-Michel Python | 4      | {">=1 overlap in 'websites'"}                              |

## License

`ms_blocking` was created by RTiedrez. It is licensed under the terms of the GNU General Public License v3.0 license.


## Credits

Our work is mostly based on [`py_entitymatching`](https://github.com/anhaidgroup/py_entitymatching)'s blocking features. Our aim was to create a fast, lightweight, portable, and RAM-preserving interface to said features, which we accomplished by redesigning `ms_blocking` as a wrapper around `pandas`, `scipy`, and `networkx` methods.