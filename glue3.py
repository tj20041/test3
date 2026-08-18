import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
logger = glueContext.get_logger()
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define schema for input invoices.
# NOTE: 'item_amounts' is ARRAY<DOUBLE>. Per-row summation of array elements
# MUST use F.aggregate() (a higher-order array function), NOT F.sum().
# F.sum() is a grouped/windowed aggregate that requires a scalar NUMERIC column
# and will raise DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE on an ARRAY column.
schema = StructType([
    StructField("invoice_id", StringType(), True),
    StructField("item_amounts", ArrayType(DoubleType()), True),
])

data = [
    ("INV-9001", [120.50, 45.00, 19.99]),
    ("INV-9002", [500.00, 150.25]),
]

invoices_df = spark.createDataFrame(data, schema)

# Log the resolved schema to CloudWatch for audit and early type-mismatch detection.
logger.info("invoices_df schema: " + invoices_df.schema.simpleString())

try:
    # Calculate total invoice amount across line items using F.aggregate().
    # F.aggregate(col, zero_value, merge_func) folds the ARRAY<DOUBLE> elements
    # into a single DOUBLE scalar per row — the correct per-row array summation
    # pattern for use inside withColumn().
    processed_invoices = invoices_df.withColumn(
        "total_invoice_amount",
        F.aggregate(
            F.col("item_amounts"),
            F.lit(0.0).cast("double"),
            lambda acc, x: acc + x
        )
    )

    # Log the output schema to CloudWatch for audit trail.
    logger.info("processed_invoices schema: " + processed_invoices.schema.simpleString())

    # Process invoice dataset.
    processed_invoices.collect()

except Exception as e:
    logger.error("Job failed with exception: " + str(e))
    raise
finally:
    # Always commit to keep the Glue job bookmark in a consistent state,
    # regardless of whether the job succeeded or failed.
    job.commit()
