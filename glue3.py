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

# Assert schema correctness at startup to catch regressions early
assert dict(invoices_df.dtypes)['item_amounts'] == 'array<double>', \
    'Schema mismatch: item_amounts must be array<double>'

# Calculate total invoice amount across line items within each row.
# F.aggregate() reduces array elements within a single row using the provided
# initial accumulator value and a binary merge lambda — no WindowSpec required.
# F.sum() is a cross-row aggregate and cannot be used here against an ArrayType column.
processed_invoices = invoices_df.withColumn(
    "total_invoice_amount",
    F.aggregate(
        F.col("item_amounts"),
        F.lit(0.0).cast(DoubleType()),
        lambda acc, x: acc + x
    )
)

# Write output to S3 using the Glue DynamicFrame writer pattern.
# collect() is intentionally replaced to avoid driver OOM on large datasets.
processed_invoices.write.mode("overwrite").parquet("s3://your-bucket/output/processed_invoices/")

job.commit()
