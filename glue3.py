import sys
import logging
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

# ==========================================
# 0. GLUE INITIALIZATION & LOGGING SETUP
# ==========================================
# Fetch job name passed by the AWS Glue execution environment
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger = logging.getLogger("FinancialAnalyticsETL")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if re-run
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.info("Initializing Financial Transactions Analytics ETL Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    transactions_schema = StructType([
        StructField("transaction_id", IntegerType(), True),
        StructField("account_id", IntegerType(), True),
        StructField("raw_amount", StringType(), True),
        StructField("fee_rate", DoubleType(), True),
        StructField("transaction_type", StringType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting transaction data into Bronze layer...")
    transactions_bronze = spark.createDataFrame([
        (1001, 501, "1250.75", 0.02, "SETTLED"),
        (1002, 502, "300.00", 0.015, "SETTLED"),
        (1003, 503, "45.50", 0.03, "PENDING"),
        (1004, 501, "890.20", 0.02, "SETTLED")
    ], schema=transactions_schema)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering settled transactions for Silver layer...")
    transactions_silver = transactions_bronze.filter("transaction_type = 'SETTLED'")

    # ==========================================
    # 4. GOLD LAYER (Financial Metric Calculation)
    # ==========================================
    logger.info("Calculating net settlement amounts for Gold layer...")

    # SPECIFIC ERROR: Type mismatch in arithmetic operation
    # 'raw_amount' is of StringType. Performing multiplication with numeric (1.0 - col("fee_rate"))
    # causes PySpark to throw an AnalysisException due to incompatible data types for arithmetic.
    # FIX: Explicitly cast 'raw_amount' to DoubleType:
    # transactions_silver.withColumn("net_amount", col("raw_amount").cast(DoubleType()) * (1.0 - col("fee_rate")))
    df_gold = transactions_silver.withColumn(
        "net_amount",
        col("raw_amount") * (1.0 - col("fee_rate"))
    ).select(
        "transaction_id",
        "account_id",
        "raw_amount",
        "net_amount"
    )

    logger.info("Pipeline completed successfully.")
    df_gold.show(truncate=False)

    # Commit Glue Job
    job.commit()

except Exception as e:
    # Error handling and logging for production diagnostics
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
