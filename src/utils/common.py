from pathlib import Path
from typing import List, Union

from loguru import logger
from pyspark.sql import SparkSession


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
