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

# Calculate total invoice amount across line items.
# F.sum() is a row-wise aggregate for scalar columns and cannot operate on
# ArrayType columns. F.aggregate() is the correct PySpark higher-order
# function for folding (summing) all elements within an array in a single row.
# F.coalesce guards against null arrays; the inner F.coalesce guards against
# null elements within the array, ensuring no NullPointerException at runtime.
processed_invoices = invoices_df.withColumn(
    "total_invoice_amount",
    F.aggregate(
        F.coalesce(F.col("item_amounts"), F.array().cast(ArrayType(DoubleType()))),
        F.lit(0.0).cast(DoubleType()),
        lambda acc, x: acc + F.coalesce(x, F.lit(0.0).cast(DoubleType()))
    )
)

# Process invoice dataset
# NOTE: collect() is used here for small test datasets only.
# For production use, replace with a write to S3 or Glue Data Catalog, e.g.:
#   processed_invoices.write.parquet('s3://your-bucket/processed_invoices/')
processed_invoices.collect()

job.commit()
