from functools import reduce
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    avg,
    coalesce,
    col,
    collect_set,
    concat_ws,
    countDistinct,
)
from pyspark.sql.functions import max as pyspark_max
from pyspark.sql.functions import min, row_number, sum
from pyspark.sql.types import (
    ByteType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
)
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


# Function to calculate memory usage in MB
def calculate_memory_usage(df):
    return df.rdd.map(lambda row: row.__sizeof__()).sum() / 1024**2


def reduce_mem_usage(df: DataFrame) -> DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage
    """

    start_mem = calculate_memory_usage(df)
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col_name, col_type in df.dtypes:
        if (
            col_type == "int"
            or col_type == "bigint"
            or col_type == "tinyint"
            or col_type == "smallint"
        ):
            min_val, max_val = df.select(
                F.coalesce(F.min(col_name), F.lit(0)),
                F.coalesce(F.max(col_name), F.lit(0)),
            ).first()
            if min_val is not None and max_val is not None:
                if min_val > -128 and max_val < 127:
                    df = df.withColumn(col_name, df[col_name].cast(ByteType()))
                elif min_val > -32768 and max_val < 32767:
                    df = df.withColumn(col_name, df[col_name].cast(ShortType()))
                elif min_val > -2147483648 and max_val < 2147483647:
                    df = df.withColumn(col_name, df[col_name].cast(IntegerType()))
                else:
                    df = df.withColumn(col_name, df[col_name].cast(LongType()))
        elif col_type == "double" or col_type == "float":
            min_val, max_val = df.select(
                F.coalesce(F.min(col_name), F.lit(0.0)),
                F.coalesce(F.max(col_name), F.lit(0.0)),
            ).first()
            if min_val is not None and max_val is not None:
                if min_val > -3.4e38 and max_val < 3.4e38:
                    df = df.withColumn(col_name, df[col_name].cast(FloatType()))
                else:
                    df = df.withColumn(col_name, df[col_name].cast(DoubleType()))
        elif col_type == "string":
            df = df.withColumn(col_name, df[col_name].cast(StringType()))

    end_mem = calculate_memory_usage(df)
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

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


def get_max_per_group(
    df: DataFrame,
    group_by_column: str,
    join_indicator: bool = False,
    columns_to_skip: List[str] = None,
):
    """
    Get the maximum value of all columns in the DataFrame, grouped by a specified column.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        group_by_column (str): The column to group by.

    Returns:
        DataFrame: A DataFrame with the maximum values for each group.
    """
    default_columns_to_skip = [group_by_column, "num_group1", "num_group2", "num_group"]
    if columns_to_skip:
        columns_to_skip = columns_to_skip + default_columns_to_skip
    else:
        columns_to_skip = default_columns_to_skip

    # Select numeric columns for the max operation
    numeric_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, (IntegerType, DoubleType, FloatType, LongType))
        and field.name not in columns_to_skip
    ]

    # Apply the max function to all numeric columns
    max_exprs = [pyspark_max(col).alias(f"Max_{col}") for col in numeric_cols]
    grouped_max_train_applprev_df = df.groupBy(group_by_column).agg(*max_exprs)

    grouped_max_train_applprev_df = grouped_max_train_applprev_df.withColumnRenamed(
        "Max_case_id", "case_id"
    )

    if join_indicator:
        return df.join(grouped_max_train_applprev_df, on="case_id", how="left")

    return grouped_max_train_applprev_df


def clean_previous_applications_dataset(df):
    df.sort("case_id", "num_group1")

    df = set_table_dtypes(df)

    missing_values_dict = get_columns_high_missing_values(
        df, threshold=10, display=False
    )
    df = df.drop(*list(missing_values_dict.keys()))

    df = get_max_per_group(df=df, group_by_column="case_id", join_indicator=True)

    window_spec = Window.partitionBy("case_id").orderBy(col("creationdate_885D").desc())
    df = df.withColumn("row_number", row_number().over(window_spec))
    df = df.filter(col("row_number") == 1).drop("row_number")

    df = drop_columns_with_few_unique_values(df)

    return df


def process_group_credit_bureau_dataset(df, num_group, threshold):
    df_group = df.filter(df.num_group1 == num_group)
    missing_values_dict = get_columns_high_missing_values(
        df_group, threshold=threshold, display=False
    )
    df_group = df_group.drop(*list(missing_values_dict.keys()))
    logger.debug(
        f"Group {num_group}: Total Number of Rows: {df_group.count()}, "
        f"Total Number of Unique Case ID: {df_group.select(countDistinct('case_id')).collect()[0][0]}"
    )
    new_columns = [
        f"{column_name} as {column_name}_num_group_{num_group}"
        for column_name in df_group.columns
    ]
    return df_group.selectExpr(*new_columns)


def clean_credit_bureau_dataset(df):
    df = set_table_dtypes(df)

    df_credit_bureau_num_group_0 = process_group_credit_bureau_dataset(df, 0, 5)
    df_credit_bureau_num_group_1 = process_group_credit_bureau_dataset(df, 1, 5)

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

    df = get_max_per_group(df=df, group_by_column="case_id", join_indicator=True)

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
        pyspark_max("amount").alias("max_amount"),
        min("tax_registry_date").alias("min_tax_registry_date"),
        pyspark_max("tax_registry_date").alias("max_tax_registry_date"),
        concat_ws(", ", collect_set("tax_registry_provider")).alias(
            "tax_registry_provider"
        ),
    )
    return df


def row_missing_value_analysis(df, display_histogram=True, title_text=""):
    null_columns = [col(column).isNull().cast("int") for column in df.columns]

    # Sum the null value indicators across the columns for each row using reduce
    missing_count_col = reduce(lambda a, b: a + b, null_columns)

    # Add the missing_count column to the DataFrame
    df_missing_count = df.withColumn("missing_count", missing_count_col).select(
        "case_id", "WEEK_NUM", "missing_count"
    )

    if display_histogram:
        # Collect the data from the DataFrame
        missing_count_data = df_missing_count.select("missing_count").collect()

        # Convert to a list of values
        missing_count_values = [row["missing_count"] for row in missing_count_data]

        # Create a histogram using Plotly
        fig = px.histogram(
            missing_count_values,
            title="Histogram of Missing Values Count - " + title_text,
            labels={"value": "Missing Count", "count": "Frequency"},
        )

        # Show the plot
        fig.show()
