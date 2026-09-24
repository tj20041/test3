import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Define schema for input invoices
schema = StructType([
    StructField("invoice_id", StringType(), True),
    StructField("item_amounts", ArrayType(DoubleType()), True),
])

data = [
    ("INV-9001", [120.50, 45.00, 19.99]),
    ("INV-9002", [500.00, 150.25]),
]

invoices_df = spark.createDataFrame(data, schema)

# Schema validation guard: ensure 'item_amounts' is an ArrayType before
# applying any per-row array reduction. This prevents future type-mismatch
# regressions (e.g. someone accidentally calling a grouped aggregate
# function such as F.sum() directly on this column) from reaching
# job.commit() with an unclassified Catalyst AnalysisException.
item_amounts_field = next(
    (f for f in invoices_df.schema.fields if f.name == "item_amounts"),
    None,
)
if item_amounts_field is None or not isinstance(item_amounts_field.dataType, ArrayType):
    raise ValueError(
        "Expected 'item_amounts' column to be ArrayType, got: "
        f"{item_amounts_field.dataType if item_amounts_field else 'MISSING'}"
    )

try:
    # Calculate total invoice amount across line items.
    # F.sum() is a grouped-aggregate function and cannot be applied
    # row-wise directly to an ArrayType(DoubleType()) column (this was
    # the root cause of the AnalysisException / GEN-UNCLASSIFIED-ERROR).
    # Use the higher-order `aggregate` SQL function to reduce each row's
    # array of item amounts into a single scalar total.
    processed_invoices = invoices_df.withColumn(
        "total_invoice_amount",
        F.expr("aggregate(item_amounts, CAST(0 AS DOUBLE), (acc, x) -> acc + x)"),
    )

    # Log the resulting schema to CloudWatch so any future type-mismatch
    # regressions surface a clear, classified error before job.commit().
    logger.info("processed_invoices schema: %s", processed_invoices.schema.simpleString())
    processed_invoices.printSchema()
except Exception as e:
    logger.error("Failed to compute total_invoice_amount: %s", str(e))
    raise

# Process invoice dataset
processed_invoices.collect()

job.commit()
