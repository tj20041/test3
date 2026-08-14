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

# Define schema for input invoices
schema = StructType([
    StructField("invoice_id", StringType(), True),
    StructField("item_amounts", ArrayType(DoubleType()), True),
])

data = [
    ("INV-9001", [120.50, 45.00, 19.99]),
    ("INV-9002", [500.00, 150.25]),
]

try:
    invoices_df = spark.createDataFrame(data, schema)

    # Calculate total invoice amount across line items using F.aggregate
    # F.sum() is a row-reducing aggregate and cannot sum elements within an array.
    # F.aggregate() correctly reduces each row's ArrayType(DoubleType()) column
    # to a scalar sum without collapsing rows, which is the intended behaviour.
    processed_invoices = invoices_df.withColumn(
        "total_invoice_amount",
        F.aggregate(
            F.col("item_amounts"),
            F.lit(0.0).cast("double"),
            lambda acc, x: acc + x
        )
    )

    # Data-quality check: warn if any invoice has a null or zero total
    zero_or_null_count = processed_invoices.filter(
        F.col("total_invoice_amount").isNull() | (F.col("total_invoice_amount") == 0.0)
    ).count()
    if zero_or_null_count > 0:
        logger.warn(
            f"DATA QUALITY WARNING: {zero_or_null_count} invoice(s) have a null or zero "
            "total_invoice_amount. Check upstream item_amounts data."
        )

    # Write results to S3 as Parquet instead of collect() to avoid driver OOM
    # on large invoice datasets in production. Replace the bucket path as needed.
    processed_invoices.write.mode("overwrite").parquet(
        "s3://your-bucket/processed-invoices/"
    )

    logger.info("Invoice processing completed successfully.")
    job.commit()

except Exception as e:
    logger.error(f"Glue job failed with exception: {str(e)}")
    raise
