import pandas as pd
import sankey as sk


def load_data(filepath):
    """
    Loads data from a JSON file into a pandas DataFrame.

    Parameters:
    filepath (str): Path to the JSON file containing artist data.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    df = pd.read_json(filepath, orient='columns')
    return df


def aggregate_data(df, COLS, col1, col2, col3):
    """
    Aggregates data by counting artists grouped by Nationality, Gender, and Decade.
    This prevents filtering out 0-decade values too early, ensuring that all data is considered.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing artist data.
    COLS (list): List of column names to group by (Nationality, Gender, BeginDate).
    col1 (str): Column representing the original year (BeginDate).
    col2 (str): New column to store decade values.
    col3 (str): New column name for storing the number of artists.

    Returns:
    pd.DataFrame: An aggregated DataFrame with artist counts grouped by the specified columns.
    """
    df[col1] = df[col1].fillna(0).astype(int)  # Convert BeginDate to integer, replacing NaNs with 0
    df[col2] = (df[col1] // 10) * 10  # Convert BeginDate into Decade (rounded down to nearest 10)
    COLS[2] = col2  # Replace 'BeginDate' in COLS with 'Decade'
    # Group by Nationality, Gender, and Decade, counting occurrences
    aggregated_df = df.groupby(COLS).size().reset_index(name=col3)
    return aggregated_df


def clean_data(df, threshold_num, col1, col2):
    """
    Cleans the aggregated DataFrame by removing:
    - Rows where `Decade == 0` (unknown dates).
    - Rows with missing values.
    - Rows where the artist count is below the given threshold.

    Parameters:
    df (pd.DataFrame): Aggregated DataFrame.
    threshold_num (int): Minimum threshold for artist counts.
    col1 (str): Column representing decade values.
    col2 (str): Column representing the count of artists.

    Returns:
    pd.DataFrame: A cleaned DataFrame with only meaningful data.
    """
    df = df[df[col1] > 0]  # Remove unknown decades (Decade == 0)
    df.dropna(inplace=True)  # Remove rows with missing values
    df = df[df[col2] >= threshold_num]  # Keep only rows with artist count above the threshold
    return df


def main():
    COLS = ["Nationality", "Gender", "BeginDate"]  # Columns to be used for aggregation
    FILE_PATH = "/Users/mac/Documents/courses/DS3500/hw/hw2/artists.json"  # File path to the dataset
    beg_COL = "BeginDate"  # Original artist start date column
    dec_COL = "Decade"  # New column for grouping data by decade
    num_COL = "num_artists"  # Column name for the aggregated count of artists
    # Step 1: Load Data
    df = load_data(FILE_PATH)
    # Step 2: Aggregate Data (counting artists per category)
    df1 = aggregate_data(df, COLS, beg_COL, dec_COL, num_COL)
    # Step 3: Clean Data (remove NaNs, filter out low-count categories)
    df2 = clean_data(df1, 20, dec_COL, num_COL)
    # Step 4: Generate Sankey Diagram
    sk.make_sankey(df2, ["Nationality", "Gender", "Decade"], vals=num_COL)
if __name__ == "__main__":
    main()
