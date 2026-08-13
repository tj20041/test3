import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

df_invoices = spark.read.json("s3://source-bucket/invoices_2026/")

invoices_with_total = df_invoices.withColumn(
    "invoice_total",
    col("line_items.unit_price") * col("line_items.quantity")
)

invoices_with_total.write.mode("overwrite").parquet("s3://output-bucket/invoice_totals/")
job.commit()
