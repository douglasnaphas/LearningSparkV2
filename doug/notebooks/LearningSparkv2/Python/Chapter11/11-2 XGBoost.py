# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # XGBoost
# MAGIC  
# MAGIC As of Databricks Runtime 9.0 ML, there is support for distributed XGBoost training. See [release notes](https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/train-model/xgboost#distributed-training) for more information.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC
# MAGIC Let's go ahead and index all of our categorical features, and set our label to be `log(price)`.

# COMMAND ----------

from pyspark.sql.functions import log, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

filePath = "dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.withColumn("label", log(col("price"))).randomSplit([.8, .2], seed=42)

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price") & (field != "label"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
pipeline = Pipeline(stages=[stringIndexer, vecAssembler])

# COMMAND ----------

# MAGIC %md ### Pyspark Distributed XGBoost
# MAGIC
# MAGIC Let's create our distributed XGBoost model. While technically not part of MLlib, you can integrate [XGBoost](https://databricks.github.io/spark-deep-learning/_modules/sparkdl/xgboost/xgboost.html) into your ML Pipelines. 
# MAGIC
# MAGIC To use the distributed version of Pyspark XGBoost you can specify two additional parameters:
# MAGIC
# MAGIC * `num_workers`: The number of workers to distribute over. Requires MLR 9.0+.
# MAGIC * `use_gpu`: Enable to utilize GPU based training for faster performance (optional).
# MAGIC
# MAGIC **NOTE:** `use_gpu` requires an ML GPU runtime. Currently, at most one GPU per worker will be used when doing distributed training. 

# COMMAND ----------

from sparkdl.xgboost import XgboostRegressor
from pyspark.ml import Pipeline

params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42, "missing": 0}

xgboost = XgboostRegressor(**params)

pipeline = Pipeline(stages=[stringIndexer, vecAssembler, xgboost])
pipeline_model = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md ## Evaluate
# MAGIC
# MAGIC Now we can evaluate how well our XGBoost model performed. Don't forget to exponentiate!

# COMMAND ----------

from pyspark.sql.functions import exp, col

log_pred_df = pipeline_model.transform(testDF)

exp_xgboost_df = log_pred_df.withColumn("prediction", exp(col("prediction")))

display(exp_xgboost_df.select("price", "prediction"))

# COMMAND ----------

# MAGIC %md Compute some metrics.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(exp_xgboost_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(exp_xgboost_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")
