import pandas as pd
import plotly.express as px
from pyspark.sql.functions import col, countDistinct
from tqdm import tqdm
import plotly.graph_objects as go


features_definitions_df = pd.read_csv(
    "raw_main_dataset/home-credit-credit-risk-model-stability/feature_definitions.csv"
)


def get_column_description(column_name):
    "Get description of the column"
    print("Column Name: ", column_name)

    row_df = features_definitions_df[
        features_definitions_df["Variable"].str.contains(column_name)
    ]
    if row_df.shape[0] == 0:
        print("Description - Not Found")
    elif row_df.shape[0] == 1:
        print(
            "Column Description: ",
            features_definitions_df.iloc[row_df.index[0]]["Description"],
        )
    else:
        raise MultipleDescriptions(
            "More than one description rows for Column: " + column_name
        )
    print("-" * 100, "\n\n")


class MultipleDescriptions(Exception):
    "Raise Exception when multiple descriptions are present for a given column"


def column_stats(df, column_name):
    "To get basic information about a column"

    # Maximum and Minimum of the column
    max_value = df.agg({column_name: "max"}).collect()[0][0]
    min_value = df.agg({column_name: "min"}).collect()[0][0]

    # Count of missing values and its percentage
    total_count = df.count()
    missing_count = df.filter(col(column_name).isNull()).count()
    missing_percentage = (missing_count / total_count) * 100

    # Number of unique values
    unique_count = df.select(countDistinct(column_name)).collect()[0][0]

    # Print results
    print(f"Column Name: {column_name}")
    print(f"Minimum Value: {min_value}")
    print(f"Maximum Value: {max_value}")
    print(f"Missing Values Count: {missing_count} ({missing_percentage:.2f}%)")
    print(f"Number of Unique Values: {unique_count}")
    print(f"Total Number of Rows: {total_count}")

    get_column_description(column_name)
    return "Function - column_stats - Done"


def create_pie_chart(df, column_name):
    "Create a Pie Chart of a single column using Plotly"

    print(f"\nCreating pie chart for '{column_name}' column:")

    # Get the counts of each unique value in the column
    value_counts_df = df.groupBy(column_name).count()

    # Convert to pandas DataFrame for plotting
    pandas_df = value_counts_df.toPandas()

    # Create a pie chart using Plotly
    fig = px.pie(
        pandas_df,
        names=column_name,
        values="count",
        title=f"Distribution of '{column_name}'",
    )

    # Update the layout for better visualization
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=True, width=500)
    fig.show()


def get_shape_df(df):
    "To get the shape of Pyspark dataframe."

    # Get the number of rows
    num_rows = df.count()

    # Get the number of columns
    num_columns = len(df.columns)

    # Print the shape of the DataFrame
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")

    # Alternatively, you can return the shape as a tuple
    shape = (num_rows, num_columns)
    print(f"Shape of DataFrame: {shape}")


def get_columns_high_missing_values(df, threshold=None, display=True):
    """
    Function to deal with Missing missing is the Dataframe:
        1. Get percentage of missing values
        2. Applying threshold ard find column with high missing values based on given threshold.
        3. Display bar-plot.
    """

    # Count of missing values and its percentage
    missing_values_dict = {}
    total_count = df.count()

    for column_name in tqdm(df.columns):
        missing_count = df.filter(col(column_name).isNull()).count()
        missing_percentage = (missing_count / total_count) * 100

        missing_values_dict[column_name] = missing_percentage

    if threshold:
        # Filter the columns based on the threshold
        missing_values_dict = {
            k: v for k, v in missing_values_dict.items() if v > threshold
        }

    if not display:
        return missing_values_dict

    # Sort the dictionary by values (percentage of missing values) in descending order
    missing_values_dict = {
        k: v
        for k, v in sorted(
            missing_values_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    # Create the bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(missing_values_dict.keys()),
            y=list(missing_values_dict.values()),
            # marker=dict(color='rgba(55, 83, 109, 0.7)', line=dict(color='rgba(55, 83, 109, 1.0)', width=1.5)),
            # opacity=0.6
        )
    )

    # Add titles and labels
    fig.update_layout(
        title="Percentage of Missing Values by Column",
        xaxis=dict(title="Columns", tickmode="linear"),
        yaxis=dict(title="Percentage of Missing Values"),
        bargap=0.2,
        width=1800,
        height=600,
    )

    # Show the plot
    fig.show()

    return missing_values_dict
