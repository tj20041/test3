import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

# Read JOB_NAME and output_path as Glue job parameters.
# output_path must be supplied as --output_path in the Glue job configuration
# (e.g. s3://your-real-bucket/output/processed_invoices/).
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_path'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Retrieve the output path from job parameters instead of hardcoding it.
output_path = args['output_path']

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

# Calculate total invoice amount across line items.
# F.sum() is a grouped aggregate function and cannot sum elements within a
# single array column row-by-row.  The correct Spark 3.x higher-order function
# is F.aggregate(), which iterates over array elements within each row and
# accumulates them into a scalar value using the supplied merge lambda.
# F.aggregate() is available on AWS Glue 3.0+ (Spark 3.x).
processed_invoices = invoices_df.withColumn(
    "total_invoice_amount",
    F.aggregate(
        F.col("item_amounts"),
        F.lit(0.0).cast(DoubleType()),
        lambda acc, x: acc + x
    )
)

# Process invoice dataset
processed_invoices.collect()

# Write output to S3 using the parameterised output path.
# Wrap in try/except so that any S3 failure (e.g. IAM 403) is logged clearly
# and the Glue job is correctly marked FAILED rather than silently succeeding.
try:
    processed_invoices.write.mode("overwrite").parquet(output_path)
except Exception as e:
    print(f"ERROR: Failed to write output to S3 path '{output_path}': {e}")
    raise

job.commit()
