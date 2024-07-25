import pickle
from pathlib import Path
from typing import List, Union

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from tqdm import tqdm


def clear_gc():
    """
    Clears the garbage collected by the Python interpreter to release memory resources.
    """
    import gc

    return gc.collect()


def import_parquet_file(
    file_path: Union[str, Path], app_name: str = "Import_Parquet_File"
):
    """Import a single Parquet File.

    Args:
        file_path (str): Path of Parquet File
        app_name (str, optional): Unique Identifiable Name for Parquet file. Defaults to "Import_Parquet_File".
    """

    if not app_name or app_name == "Import_Parquet_File":
        logger.warning(
            "WARNING: Using a default or generic app_name. It's recommended to provide a unique and descriptive app_name for better identification and tracking."
        )

    file_path = str(file_path) if isinstance(file_path, Path) else file_path

    try:
        logger.info(f"Starting Spark session with app name: {app_name}")
        # spark = SparkSession.builder.appName(app_name).getOrCreate()
        spark = SparkSession.builder.appName(app_name).getOrCreate()

        logger.info(f"Reading Parquet file from path: {file_path}")
        df = spark.read.parquet(file_path)

        logger.info("Parquet file loaded successfully")
        return df, spark
    except Exception as e:
        logger.error(f"Error: {e}")
        spark.stop()
        raise


def import_and_concatenate_parquet_files(
    base_path: Path,
    data_file_names: List[str],
    app_name: str = "Import_Multiple_Parquet_Files_Concatenate",
):
    logger.info("Initializing SparkSession")
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    if not app_name or app_name == "Import_Multiple_Parquet_Files_Concatenate":
        logger.warning(
            "WARNING: Using a default or generic app_name. It's recommended to provide a unique and descriptive app_name for better identification and tracking."
        )

    dataframes = []
    schemas = set()

    for file_name in data_file_names:
        file_path = base_path / Path(file_name)
        logger.info(f"Reading file {file_name} with app name {app_name}")
        df = spark.read.parquet(str(file_path))
        logger.info("File Loaded - Rows Count: " + str(df.count()))
        # Add schema to the set
        schemas.add(tuple(df.schema))

        # If more than one unique schema is found, raise an error
        if len(schemas) > 1:
            logger.error(
                f"Schema mismatch detected in file {file_name}. All files must have the same schema."
            )
            raise ValueError(
                f"Schema mismatch detected in file {file_name}. All files must have the same schema."
            )

        dataframes.append(df)

    if not dataframes:
        logger.error("No dataframes to concatenate. Check your input data.")
        raise ValueError("No dataframes to concatenate. Check your input data.")

    # Concatenate all DataFrames
    logger.info("Concatenating DataFrames")
    concatenated_df = dataframes[0]
    for df in dataframes[1:]:
        concatenated_df = concatenated_df.unionByName(df)

    logger.info("Concatenation complete")
    return concatenated_df, spark


def count_unique_values(df: DataFrame) -> dict:
    """
    Count the number of unique values in each column of a PySpark DataFrame.

    :param df: PySpark DataFrame
    :return: Dictionary with column names as keys and number of unique values as values
    """
    unique_counts = {}

    for column in tqdm(df.columns):
        # Get the number of distinct values in each column
        unique_count = df.select(column).distinct().count()
        unique_counts[column] = unique_count

    return unique_counts


def load_pickle(file_path):
    """
    Load data from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The object stored in the pickle file.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
    except (FileNotFoundError, pickle.PickleError) as e:
        logger.error(f"Error loading pickle file: {e}")
        raise


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
