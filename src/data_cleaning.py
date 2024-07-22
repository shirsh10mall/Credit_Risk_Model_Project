from functools import reduce

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.functions import (avg, coalesce, col, collect_set, concat_ws,
                                   countDistinct, max, min, row_number, sum)
from pyspark.sql.types import DateType, FloatType, IntegerType, StringType
from pyspark.sql.window import Window
from tqdm import tqdm

# Define mapping for column names and suffixes to data types
dtype_mapping = {
    "case_id": IntegerType(),
    "WEEK_NUM": IntegerType(),
    "num_group1": IntegerType(),
    "num_group2": IntegerType(),
    "date_decision": DateType(),
}
# Define suffix mappings
suffix_mapping = {
    "P": FloatType(),
    "A": FloatType(),
    "M": StringType(),
    "D": DateType(),
}


def set_table_dtypes(df: DataFrame) -> DataFrame:
    """
    Casts columns of the input DataFrame to specified data types based on predefined mappings.

    Args:
    df (DataFrame): Input DataFrame with columns to be cast.

    Returns:
    DataFrame: Transformed DataFrame with columns cast to specified data types.
    """
    transformations = []

    for column in df.columns:
        if column in dtype_mapping:
            transformations.append((column, dtype_mapping[column]))
        elif column.endswith(tuple(suffix_mapping.keys())):
            transformations.append((column, suffix_mapping[column[-1]]))

    for column, dtype in transformations:
        df = df.withColumn(column, col(column).cast(dtype))

    return df


def drop_columns_with_few_unique_values(df: DataFrame, threshold: int = 1) -> DataFrame:
    """Drop Columns haivng few unique values

    Args:
        df (DataFrame): [description]
        threshold (int, optional): [description]. Defaults to 2.

    Returns:
        DataFrame: [description]
    """

    # Get the list of columns to drop
    columns_to_drop = [
        col_name
        for col_name in df.columns
        if df.select(col_name).distinct().count() < threshold
    ]

    # Log the columns being dropped
    if columns_to_drop:
        logger.info(
            f"Dropping columns with fewer than {threshold} unique values: {columns_to_drop}"
        )
    else:
        logger.info(f"No columns with fewer than {threshold} unique values found.")

    # Drop the columns
    df = df.drop(*columns_to_drop)
    return df


features_definitions_df = pd.read_csv(
    "raw_main_dataset/home-credit-credit-risk-model-stability/feature_definitions.csv"
)


def get_column_description(column_name):
    """
    Get description of the column

    Parameters:
    column_name (str): The name of the column to retrieve the description for.

    Raises:
    MultipleDescriptions: If more than one description is found for the column.

    Prints the column name and its description if found, or a message if not found.
    """

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
    """
    To get basic information about a column in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the column.
    column_name (str): The name of the column to retrieve information for.

    Returns:
    str: Message indicating the completion of the function.

    Prints the column name, its minimum and maximum values, count and percentage of missing values, number of unique values, and total number of rows. Also calls 'get_column_description' to retrieve the column description.
    """

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
    """
    Create a Pie Chart of a single column using Plotly

    Parameters: 
    - df: DataFrame containing the data
    - column_name: Name of the column to create the Pie Chart for

    Prints a message indicating the creation of the chart, then generates a Pie Chart using Plotly
    to visualize the distribution of values in the specified column.
    """

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
    """
    To get the shape of Pyspark dataframe.

    Parameters:
    df (DataFrame): The input Pyspark DataFrame.

    Returns:
    None
    """

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


# Functions for Cleaning Individual Column of Dataframe.
def clean_base_dataset(df):
    df = df.sort("case_id").drop("MONTH")
    df = set_table_dtypes(df)
    return df


def clean_previous_applications_dataset(df):
    df.sort("case_id", "num_group1")

    df = set_table_dtypes(df)

    missing_values_dict = get_columns_high_missing_values(
        df, threshold=10, display=False
    )
    df = df.drop(*list(missing_values_dict.keys()))

    window_spec = Window.partitionBy("case_id").orderBy(col("creationdate_885D").desc())
    df = df.withColumn("row_number", row_number().over(window_spec))
    df = df.filter(col("row_number") == 1).drop("row_number")

    df = drop_columns_with_few_unique_values(df)

    return df


def clean_credit_bureau_dataset(df):
    df = set_table_dtypes(df)

    df = df.filter((df.num_group1 == 0) | (df.num_group1 == 1))
    df = df.sort("case_id", "num_group1")

    df_credit_bureau_num_group_0 = df.filter((df.num_group1 == 0))
    missing_values_dict = get_columns_high_missing_values(
        df_credit_bureau_num_group_0, threshold=15, display=False
    )

    df_credit_bureau_num_group_0 = df_credit_bureau_num_group_0.drop(
        *list(missing_values_dict.keys())
    )

    print(
        "Total Number of Row: ",
        df_credit_bureau_num_group_0.count(),
        "\nTotal Number of Unique Case ID: ",
        df_credit_bureau_num_group_0.select(countDistinct("case_id")).collect()[0][0],
    )

    df_credit_bureau_num_group_1 = df.filter((df.num_group1 == 1))

    missing_values_dict = get_columns_high_missing_values(
        df_credit_bureau_num_group_1, threshold=10, display=False
    )

    df_credit_bureau_num_group_1 = df_credit_bureau_num_group_1.drop(
        *list(missing_values_dict.keys())
    )

    print(
        "Total Number of Row: ",
        df_credit_bureau_num_group_1.count(),
        "\nTotal Number of Unique Case ID: ",
        df_credit_bureau_num_group_1.select(countDistinct("case_id")).collect()[0][0],
    )

    # Rename Column using selectExpr to rename columns
    new_columns = [
        f"{column_name} as {column_name}_num_group_0"
        for column_name in df_credit_bureau_num_group_0.columns
    ]
    print("Number of columns: ", len(new_columns))

    df_credit_bureau_num_group_0 = df_credit_bureau_num_group_0.selectExpr(*new_columns)

    new_columns = [
        f"{column_name} as {column_name}_num_group_1"
        for column_name in df_credit_bureau_num_group_1.columns
    ]
    df_credit_bureau_num_group_1 = df_credit_bureau_num_group_1.selectExpr(*new_columns)
    print("Number of columns: ", len(new_columns))

    df_credit_bureau = df_credit_bureau_num_group_0.join(
        df_credit_bureau_num_group_1,
        df_credit_bureau_num_group_0.case_id_num_group_0
        == df_credit_bureau_num_group_1.case_id_num_group_1,
        how="outer",
    )

    df_credit_bureau = df_credit_bureau.withColumn(
        "case_id", coalesce(col("case_id_num_group_0"), col("case_id_num_group_1"))
    ).drop("case_id_num_group_0", "case_id_num_group_1")

    return df_credit_bureau


def clean_person_dataset(df):
    df = set_table_dtypes(df)

    df = df.filter(df.num_group1 == 0)

    missing_values_dict = get_columns_high_missing_values(
        df=df, threshold=20, display=False
    )

    df = df.drop(*list(missing_values_dict.keys()) + ["num_group1"])

    return df


def clean_static_dataset(df):
    df = set_table_dtypes(df)
    df = df.sort("case_id")

    missing_values_dict = get_columns_high_missing_values(
        df, threshold=20, display=False
    )

    df = df.drop(*list(missing_values_dict.keys()))

    return df


def clean_static_cb_dataset(df):
    df = set_table_dtypes(df)
    df = df.sort("case_id")

    missing_values_dict = get_columns_high_missing_values(
        df, threshold=20, display=False
    )

    df = df.drop(*list(missing_values_dict.keys()))

    return df


def clean_tax_registry_dataset(df):
    df = set_table_dtypes(df)
    df = df.sort("case_id", "tax_registry_provider", "num_group", ascending=True)

    df = df.groupBy("case_id").agg(
        # collect_set("case_id").alias("case_id"),
        avg("amount").alias("average_tax_registry_amount"),
        sum("amount").alias("total_tax_registry_amount"),
        min("tax_registry_date").alias("min_tax_registry_date"),
        max("tax_registry_date").alias("max_tax_registry_date"),
        concat_ws(", ", collect_set("tax_registry_provider")).alias(
            "tax_registry_provider"
        ),
    )

    return df


def row_missing_value_analysis(df, display_histogram=True, title_text=''):
    null_columns = [col(column).isNull().cast("int") for column in df.columns]

    # Sum the null value indicators across the columns for each row using reduce
    missing_count_col = reduce(lambda a, b: a + b, null_columns)

    # Add the missing_count column to the DataFrame
    df_missing_count = df.withColumn("missing_count", missing_count_col).select("case_id", "WEEK_NUM", "missing_count")


    if display_histogram:
        # Collect the data from the DataFrame
        missing_count_data = df_missing_count.select("missing_count").collect()

        # Convert to a list of values
        missing_count_values = [row['missing_count'] for row in missing_count_data]

        # Create a histogram using Plotly
        fig = px.histogram(missing_count_values, title="Histogram of Missing Values Count - " + title_text,
                        labels={'value': 'Missing Count', 'count': 'Frequency'})

        # Show the plot
        fig.show()