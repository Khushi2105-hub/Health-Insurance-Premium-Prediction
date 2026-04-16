# ==============================
# IMPORT LIBRARIES
# ==============================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, abs, when

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# CREATE SPARK SESSION
# ==============================

spark = SparkSession.builder.appName("InsuranceProject").getOrCreate()


# ==============================
# LOAD DATASET
# ==============================

df = spark.read.csv(
    "C:/Users/KHUSHI/Desktop/Insurance_Premium_Prediction/insurance.csv",
    header=True,
    inferSchema=True
)

print("===== DATASET =====")
df.show(5)

print("===== SCHEMA =====")
df.printSchema()


# ==============================
# CREATE POLICY TYPE (FEATURE ENGINEERING)
# ==============================

df = df.withColumn(
    "policy_type",
    when(col("bmi") < 25, "basic")
    .when((col("bmi") >= 25) & (col("bmi") < 30), "standard")
    .otherwise("premium")
)

print("===== POLICY TYPE =====")
df.select("bmi", "policy_type").show(5)


# ==============================
# CHECK MISSING VALUES
# ==============================

print("===== MISSING VALUES =====")
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

df = df.dropna()


# ==============================
# ENCODING
# ==============================

sex_indexer = StringIndexer(inputCol="sex", outputCol="sex_index")
smoker_indexer = StringIndexer(inputCol="smoker", outputCol="smoker_index")
region_indexer = StringIndexer(inputCol="region", outputCol="region_index")
policy_indexer = StringIndexer(inputCol="policy_type", outputCol="policy_index")

encoder = OneHotEncoder(
    inputCols=["sex_index", "smoker_index", "region_index", "policy_index"],
    outputCols=["sex_vec", "smoker_vec", "region_vec", "policy_vec"]
)


# ==============================
# FEATURE ENGINEERING
# ==============================

assembler = VectorAssembler(
    inputCols=[
        "age",
        "bmi",
        "children",
        "sex_vec",
        "smoker_vec",
        "region_vec",
        "policy_vec"
    ],
    outputCol="unscaled_features"
)

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features"
)


# ==============================
# MODEL
# ==============================

lr = LinearRegression(featuresCol="features", labelCol="charges")


# ==============================
# PIPELINE
# ==============================

pipeline = Pipeline(stages=[
    sex_indexer,
    smoker_indexer,
    region_indexer,
    policy_indexer,
    encoder,
    assembler,
    scaler,
    lr
])


# ==============================
# TRAIN TEST SPLIT
# ==============================

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)


# ==============================
# TRAIN MODEL
# ==============================

model = pipeline.fit(train_data)


# ==============================
# MODEL SUMMARY
# ==============================

lr_model = model.stages[-1]

print("===== MODEL SUMMARY =====")
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)


# ==============================
# MAKE PREDICTIONS
# ==============================

predictions = model.transform(test_data)

print("===== PREDICTIONS =====")
predictions.select("age", "bmi", "policy_type", "charges", "prediction").show(10)


# ==============================
# ERROR ANALYSIS
# ==============================

predictions = predictions.withColumn(
    "error",
    abs(col("charges") - col("prediction"))
)

print("===== ERROR ANALYSIS =====")
predictions.select("charges", "prediction", "error").show(10)


# ==============================
# MODEL EVALUATION
# ==============================

rmse = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="r2").evaluate(predictions)

print("===== MODEL PERFORMANCE =====")
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==============================
# AVERAGE ERROR
# ==============================

avg_error = predictions.selectExpr("avg(error)").collect()[0][0]
print("Average Error:", avg_error)


# ==============================
# CONVERT TO PANDAS FOR VISUALIZATION
# ==============================

pdf = predictions.select("charges", "prediction").toPandas()
pdf_full = df.toPandas()


# ==============================
# VISUALIZATION 1: ACTUAL VS PREDICTED
# ==============================

plt.figure()
plt.scatter(pdf["charges"], pdf["prediction"])

plt.plot(
    [pdf["charges"].min(), pdf["charges"].max()],
    [pdf["charges"].min(), pdf["charges"].max()],
    color='red'
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Premium")
plt.show()


# ==============================
# VISUALIZATION 2: CORRELATION HEATMAP
# ==============================

plt.figure()
sns.heatmap(pdf_full.corr(numeric_only=True), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()


# ==============================
# VISUALIZATION 3: SMOKER VS CHARGES
# ==============================

plt.figure()
sns.boxplot(x=pdf_full["smoker"], y=pdf_full["charges"])
plt.title("Smoker vs Insurance Charges")
plt.xlabel("Smoker")
plt.ylabel("Charges")
plt.show()


# ==============================
# VISUALIZATION 4: ERROR DISTRIBUTION
# ==============================

plt.figure()
sns.histplot(pdf["charges"] - pdf["prediction"], kde=True)
plt.title("Residual Distribution (Error)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()


# ==============================
# STOP SPARK
# ==============================

spark.stop()