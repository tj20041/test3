import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType
from pyspark.sql.window import Window
from pyspark.errors.exceptions.captured import AnalysisException as SparkAnalysisException

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

# Assert schema to catch ArrayType regressions at the earliest possible point
assert isinstance(invoices_df.schema["item_amounts"].dataType, ArrayType), \
    "Schema assertion failed: 'item_amounts' must be ArrayType"
assert isinstance(invoices_df.schema["item_amounts"].dataType.elementType, DoubleType), \
    "Schema assertion failed: 'item_amounts' element type must be DoubleType"

try:
    # Fix 1: Sum array elements within a single row using F.aggregate() (higher-order
    # function). F.sum() is a cross-row aggregate/window function and cannot operate on
    # an ArrayType column to sum elements within a single row. F.aggregate() is the
    # correct idiomatic PySpark pattern for within-row array folding on AWS Glue Spark 3.x.
    # It takes: the array column, a zero/initial accumulator typed as DoubleType() to
    # match the declared ArrayType(DoubleType()) schema, and a binary merge lambda.
    processed_invoices = invoices_df.withColumn(
        "total_invoice_amount",
        F.aggregate(
            F.col("item_amounts"),
            F.lit(0.0).cast(DoubleType()),
            lambda acc, x: acc + x
        )
    )

    # Fix 2: Deduplicate using row_number() with a fully specified WindowSpec that
    # includes BOTH partitionBy AND orderBy. PySpark's Catalyst analyser requires every
    # ranking window function (row_number, rank, dense_rank, ntile) to have a
    # deterministic ordering; without it an AnalysisException is raised immediately
    # during logical plan analysis before any Spark action executes.
    # Here we partition by invoice_id and order by invoice_id (the available ordering
    # key in this schema). In production, replace the orderBy column with the
    # business-appropriate ordering column such as an event timestamp or sequence number
    # (e.g. F.col("event_time").desc() to keep the latest record).
    window_spec = Window.partitionBy("invoice_id").orderBy(F.col("invoice_id"))

    deduplicated_df = processed_invoices.withColumn(
        "row_num",
        F.row_number().over(window_spec)
    )

    # Fix 3: Filter to keep only the first row per partition (deduplication), then
    # drop the helper rank column before writing output.
    deduplicated_df = deduplicated_df.filter(F.col("row_num") == 1).drop("row_num")

    # Fix 4: Replace driver-materialising collect() with a distributed Parquet write,
    # which is the correct AWS Glue pattern and prevents driver OOM on large datasets.
    deduplicated_df.write.mode("overwrite").parquet("/tmp/processed_invoices_output")

except SparkAnalysisException as e:
    glueContext.get_logger().error(f"Spark plan analysis failed: {e}")
    raise

job.commit()
