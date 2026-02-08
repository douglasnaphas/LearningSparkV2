// Databricks notebook source
// MAGIC
// MAGIC %md
// MAGIC # XGBoost
// MAGIC  
// MAGIC If you are not using DBR 9.1 ML LTS, you will need to install `ml.dmlc:xgboost4j-spark_2.12:1.5.0` from Maven.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Data Preparation
// MAGIC
// MAGIC Let's go ahead and index all of our categorical features, and set our label to be `log(price)`.

// COMMAND ----------

import org.apache.spark.sql.functions.log
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline

val filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
val airbnbDF = spark.read.parquet(filePath)
val Array(trainDF, testDF) = airbnbDF.withColumn("label", log($"price")).randomSplit(Array(.8, .2), seed=42)

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price" && field != "label"}.map(_._1)
val assemblerInputs = indexOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(Array(stringIndexer, vecAssembler))

// COMMAND ----------

// MAGIC %md
// MAGIC ## XGBoost
// MAGIC
// MAGIC Now we are ready to train our XGBoost model!

// COMMAND ----------


import ml.dmlc.xgboost4j.scala.spark._
import org.apache.spark.sql.functions._

val paramMap = List("num_round" -> 100, "eta" -> 0.1, "max_leaf_nodes" -> 50, "seed" -> 42, "missing" -> 0).toMap

val xgboostEstimator = new XGBoostRegressor(paramMap)

val xgboostPipeline = new Pipeline().setStages(pipeline.getStages ++ Array(xgboostEstimator))

val xgboostPipelineModel = xgboostPipeline.fit(trainDF)
val xgboostLogPredictedDF = xgboostPipelineModel.transform(testDF)

val expXgboostDF = xgboostLogPredictedDF.withColumn("prediction", exp(col("prediction")))
expXgboostDF.createOrReplaceTempView("expXgboostDF")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate
// MAGIC
// MAGIC Now we can evaluate how well our XGBoost model performed.

// COMMAND ----------

val expXgboostDF = spark.table("expXgboostDF")

display(expXgboostDF.select("price", "prediction"))

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val regressionEvaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = regressionEvaluator.evaluate(expXgboostDF)
val r2 = regressionEvaluator.setMetricName("r2").evaluate(expXgboostDF)
println(s"RMSE is $rmse")
println(s"R2 is $r2")
println("*-"*80)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Export to Python
// MAGIC
// MAGIC We can also export our XGBoost model to use in Python for fast inference on small datasets.

// COMMAND ----------


val nativeModelPath = "xgboost_native_model"
val xgboostModel = xgboostPipelineModel.stages.last.asInstanceOf[XGBoostRegressionModel]
xgboostModel.nativeBooster.saveModel(nativeModelPath)
