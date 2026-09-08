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

# Validate that item_amounts column is of the expected array<double> type before transformation
assert dict(invoices_df.dtypes).get("item_amounts") == "array<double>", \
    "Schema mismatch: item_amounts must be array<double>"

# Calculate total invoice amount across line items using F.aggregate() to
# perform an element-wise sum over the array within each row.
# F.sum() is a row-wise aggregation function for scalar numeric columns and
# cannot reduce elements within an ARRAY<DOUBLE> column — use F.aggregate() instead.
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

job.commit()
