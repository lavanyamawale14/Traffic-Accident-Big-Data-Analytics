
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

# 1. Initialize Apache Spark
spark = (
    SparkSession.builder
    .appName("Indian Road Accident Analysis")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. Load CSV dataset
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "indian_roads_dataset.csv"

df = spark.read.csv(
    str(DATASET_PATH),
    header=True,
    inferSchema=True
)

# 3. Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Total Records:", df.count())
print("Total Columns:", len(df.columns))

print("\n========== DATASET SCHEMA ==========")
df.printSchema()

print("\n========== FIRST 5 RECORDS ==========")
df.show(5, truncate=False)

# 4. Accident severity distribution
print("\n========== ACCIDENT SEVERITY ==========")

df.groupBy("accident_severity") \
    .agg(count("*").alias("total_accidents")) \
    .orderBy(desc("total_accidents")) \
    .show()

# 5. State-wise accident analysis
print("\n========== STATE-WISE ACCIDENTS ==========")

df.groupBy("state") \
    .agg(count("*").alias("total_accidents")) \
    .orderBy(desc("total_accidents")) \
    .show()

# 6. Weather-wise accident analysis
print("\n========== WEATHER ANALYSIS ==========")

df.groupBy("weather") \
    .agg(count("*").alias("total_accidents")) \
    .orderBy(desc("total_accidents")) \
    .show()

# 7. Accident hotspot detection
print("\n========== TOP 10 ACCIDENT HOTSPOTS ==========")

df.groupBy("city", "state") \
    .agg(count("*").alias("total_accidents")) \
    .orderBy(desc("total_accidents")) \
    .show(10, truncate=False)

spark.stop()
