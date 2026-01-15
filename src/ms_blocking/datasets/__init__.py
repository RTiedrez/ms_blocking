from pandas import read_csv
from importlib import resources


def get_users():
    """Get example data for testing the functionalities of the package

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame that contains the example data.

    """

    data_file_path = resources.files("ms_blocking.datasets") / "user_data.csv"
    return read_csv(str(data_file_path))
