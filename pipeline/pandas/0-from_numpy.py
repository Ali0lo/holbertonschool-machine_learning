import pandas as pd
def from_numpy(array):
    cols = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
import pandas as pd

def from_numpy(array):
    """Creates a pandas DataFrame from a NumPy array.

    Columns are labeled alphabetically in uppercase.

    Args:
        array (numpy.ndarray): input array

    Returns:
        pandas.DataFrame: resulting DataFrame
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
