import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio


pd.set_option('future.no_silent_downcasting', True)
pio.renderers.default = 'browser'

def _code_mapping(df, src, targ):
    """
    Converts text labels into numeric node indexes for the Sankey diagram.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing source-target relationships.
    src (str): Column representing the source nodes.
    targ (str): Column representing the target nodes.

    Returns:
    tuple: A modified DataFrame with numeric mappings and a list of unique labels.
    """
    labels = list(set(df[src]).union(set(df[targ])))  # Collect unique labels
    lc_map = dict(zip(labels, range(len(labels))))  # Map labels to unique numeric indices
    df = df.replace({src: lc_map, targ: lc_map})  # Replace text labels with numeric values
    return df, labels

def make_sankey(df, cols, vals):
    """
    Generates and displays a Sankey diagram from the input DataFrame.

    Parameters:
    df (pd.DataFrame): The dataset containing relationships between categories.
    cols (list): List of column names defining the Sankey flow (e.g., ["Nationality", "Gender", "Decade"]).
    vals (str): Column representing the values or counts for the links between nodes.

    Returns:
    None: Displays the generated Sankey diagram.
    """

    df.loc[df[vals] == 0, vals] = 1  # Ensure the thickness of the sankey is default as 1 even there are 0 artists
    # Initialize an empty DataFrame for stacked relationships
    stacked = pd.DataFrame(columns=["source", "target", "value"])
    # Iterate through column pairs to create source-target relationships
    for i in range(len(cols) - 1):
        df_1 = df.groupby([cols[i], cols[i + 1]])[vals].sum().reset_index()
        df_1.columns = ["source", "target", "value"]  # Standardize column names
        stacked = pd.concat([stacked, df_1], axis=0, ignore_index=True)  # Append results
    # Aggregate duplicate source-target pairs to sum their values
    stacked = stacked.groupby(["source", "target"], as_index=False)["value"].sum()
    # Convert categorical labels into numeric indices for Sankey diagram
    stacked, labels = _code_mapping(stacked, "source", "target")
    # Construct Sankey diagram components
    link = {'source': stacked["source"], 'target': stacked["target"], 'value': stacked["value"]}
    node = {'label': labels}  # Nodes labeled using unique text categories
    # Generate and display the Sankey diagram
    fig = go.Figure(go.Sankey(link=link, node=node))
    fig.show()
