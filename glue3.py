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

# Defensive guard: ensure item_amounts is the expected array type before transforming
assert dict(invoices_df.dtypes).get('item_amounts') == 'array<double>', \
    "item_amounts must be of type ARRAY<DOUBLE> — got: {}".format(
        dict(invoices_df.dtypes).get('item_amounts')
    )

# Calculate total invoice amount across line items.
# F.sum() is a cross-row aggregate and cannot accept ARRAY<DOUBLE>.
# F.aggregate() is the correct Spark 3.x function for folding array elements
# into a per-row scalar value.
processed_invoices = invoices_df.withColumn(
    "total_invoice_amount",
    F.aggregate(
        F.col("item_amounts"),
        F.lit(0.0).cast(DoubleType()),
        lambda acc, x: acc + x
    )
)

# Confirm output schema: total_invoice_amount must resolve to DoubleType, not ArrayType
invoices_df.printSchema()
processed_invoices.printSchema()

# Process invoice dataset
processed_invoices.collect()

job.commit()
