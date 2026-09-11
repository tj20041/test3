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

try:
    # Calculate total invoice amount across line items.
    # NOTE: pyspark.sql.functions.sum is a row-aggregation function and cannot
    # operate on an ArrayType column. To sum the elements *within* each row's
    # array we must use the higher-order SQL function `aggregate`, invoked via
    # F.expr, which reduces the array to a single scalar DoubleType value per row.
    processed_invoices = invoices_df.withColumn(
        "total_invoice_amount",
        F.expr("aggregate(item_amounts, CAST(0 AS DOUBLE), (acc, x) -> acc + x)")
    )

    # Defensive schema check to catch type-mismatch regressions early, before
    # the action below forces plan evaluation.
    processed_invoices.printSchema()

    # Force evaluation and surface the computed totals for diagnostics/logging.
    # (Kept as collect() per existing behaviour; replace with an explicit
    # write_dynamic_frame sink if this job needs to persist results downstream.)
    results = processed_invoices.collect()
    for row in results:
        logger.info(
            "Processed invoice_id=%s total_invoice_amount=%s",
            row["invoice_id"],
            row["total_invoice_amount"],
        )

    job.commit()
except Exception:
    logger.error("Glue job failed while computing total_invoice_amount.", exc_info=True)
    logger.error("invoices_df schema:")
    invoices_df.printSchema()
    raise
