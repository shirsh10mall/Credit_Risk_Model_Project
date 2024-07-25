import os
import pickle
from pathlib import Path
from typing import List

from loguru import logger
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, datediff, lit, when
from pyspark.sql.types import (BooleanType, DateType, DoubleType, FloatType,
                               IntegerType, LongType, NumericType)
from tqdm import tqdm

from src.utils.common import count_unique_values

CURRENT_DIRECTORY_NOTEBOOK = Path(os.getenv("PROJECT_BASE_PATH"))


class NumericalColumnDataTypeError(Exception):
    pass


def fill_missing_values(df: DataFrame, fill_value: any) -> DataFrame:
    """
    Identify missing values in the PySpark DataFrame and fill them with the specified value.

    :param df: PySpark DataFrame with missing values
    :param fill_value: Value to fill in missing entries
    :return: DataFrame with missing values filled
    """

    # Get a list of all columns in the DataFrame
    columns = df.columns
    df_schema = df.schema
    # Iterate over each column and fill missing values
    for column in columns:
        if df_schema[column].dataType not in [
            DoubleType(),
            FloatType(),
            IntegerType(),
            LongType(),
        ]:
            raise NumericalColumnDataTypeError(
                f"Error in fill_missing_values function. | Column Name: {column} | Data Type: {df_schema[column].dataType}"
            )
        # Apply the fill value to each column where the value is missing
        df = df.withColumn(
            column, when(col(column).isNull(), fill_value).otherwise(col(column))
        )

    return df


def process_dates(
    df: DataFrame, date_columns: list, base_date: str = "1940-01-01"
) -> DataFrame:
    """
    Process the DataFrame by setting a base date, casting date columns,
    and calculating the difference in days from the base date.

    :param df: PySpark DataFrame to process
    :param base_date: Base date string in the format 'YYYY-MM-DD'
    :param date_columns: List of column names that should be treated as dates
    :return: Processed DataFrame with updated date columns
    """

    # Add base_date column and cast it to DateType
    df = df.withColumn("base_date", lit(base_date).cast(DateType()))

    # Cast specified columns to DateType
    for column_name in tqdm(date_columns):
        df = df.withColumn(column_name, col(column_name).cast(DateType()))

    # Compute date difference for each date column
    for column_name in tqdm(date_columns):
        df = df.withColumn(column_name, datediff(col(column_name), col("base_date")))

    return df


def drop_single_unique_columns(df: DataFrame) -> DataFrame:
    """
    Drop columns with only one unique value from the PySpark DataFrame.

    :param df: PySpark DataFrame
    :return: DataFrame with columns having only one unique value removed
    """
    # Get unique value counts for each column
    unique_counts = count_unique_values(df)

    # Identify columns with only one unique value
    columns_to_drop = [column for column, count in unique_counts.items() if count == 1]

    # Log the columns to drop
    if columns_to_drop:
        logger.info(f"Columns with only one unique value to drop: {columns_to_drop}")
    else:
        logger.info("No columns with only one unique value found.")

    # Drop the identified columns
    df = df.drop(*columns_to_drop)

    return df


def check_contains_negative_value(df: DataFrame) -> bool:
    """
    Check if there is any negative value in any column of the DataFrame.

    :param df: PySpark DataFrame
    :return: Boolean indicating if there is any negative value
    """
    column_having_negative_values = []
    for column in tqdm(df.columns):
        # Check if the column is of numeric type
        if isinstance(df.schema[column].dataType, NumericType):
            # Filter rows where the column value is negative
            if df.filter(col(column) < 0).count() > 0:
                column_having_negative_values.append(column)

    return column_having_negative_values


def convert_boolean_to_integer(
    df: DataFrame, boolean_cols: List[str] = None
) -> DataFrame:
    """
    Convert boolean columns in a PySpark DataFrame to integer columns (False -> 0, True -> 1).

    :param df: PySpark DataFrame
    :return: DataFrame with boolean columns converted to integer
    """
    if not boolean_cols:
        boolean_cols = df.columns

    for column in boolean_cols:
        # Check if the column is of Boolean type
        if isinstance(df.schema[column].dataType, BooleanType):
            # Convert the column to integer
            df = df.withColumn(column, col(column).cast("integer"))

    return df


def convert_categorical_to_integers(
    df: DataFrame,
    categorical_cols: List[str],
    pickle_path: str = CURRENT_DIRECTORY_NOTEBOOK
    / Path("saved_pipeline_files/categorical_to_integers_mapper.pickle"),
) -> DataFrame:
    """
    Convert categorical columns of a PySpark DataFrame into integers and save the mapping dictionary into a pickle file.

    :param df: Input PySpark DataFrame
    :param categorical_cols: List of categorical column names to be converted
    :param pickle_path: File path to save the mapping dictionary as a pickle file
    :return: DataFrame with categorical columns converted to integers
    """
    # Dictionary to store the mapping of categorical values to integers
    mappings = {}

    for col_name in tqdm(categorical_cols):
        logger.info(f"Processing column: {col_name}")

        # StringIndexer for the current column
        indexer = StringIndexer(
            inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep"
        )

        # Fit the indexer and transform the data
        indexed_df = indexer.fit(df).transform(df)

        # Extract the mapping of the original values to the index
        mapping = dict(enumerate(indexer.fit(df).labels))
        mappings[col_name] = mapping

        # Drop the original categorical column and rename the indexed column
        df = indexed_df.drop(col_name).withColumnRenamed(f"{col_name}_index", col_name)

    # Save the mappings dictionary to a pickle file
    with open(pickle_path, "wb") as f:
        pickle.dump(mappings, f)
        logger.info(f"Saved mappings to {pickle_path}")

    return df
