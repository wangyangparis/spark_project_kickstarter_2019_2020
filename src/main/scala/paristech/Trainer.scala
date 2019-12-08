package paristech

//import java.util.logging.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.{concat_ws, datediff, from_unixtime, lower, round, second, udf}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.Logger
import org.apache.log4j.Level


object Trainer {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    //import the dataset from parquet files at "src/main/resource/prepared_trainingset/"
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet("src/main/resources/prepared_trainingset/")

     //tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //remove stopwords
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    //TF with CountVectorizer
    val tf = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    //another solution is use hashingTF
      val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol)
        .setOutputCol("rawFeatures")
        .setNumFeatures(20)

    //val featurizedData = hashingTF.transform(stopwords)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("tfidf")

    //use index for country and manage the exception
    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    //use index for currency
    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    //Use OneHotEncoder for the country index and currency index
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    //Assembler the features in a col-matrix
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    //Logistic regresstion (just for experiment)
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    //creat a pipeline with different steps
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, hashingTF, idf, indexer_country,indexer_currency,encoder,assembler,lr))

    //Model training
    val model = pipeline.fit(df)
    model.transform(df).show()

    // Split training and test data.
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 12)

    //Logistic regression that we will use
    val lr1 = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(30)

    // pipeline we actually use
    val pipeline1 = new Pipeline()
      .setStages(Array(tokenizer, remover, tf, idf, indexer_country,indexer_currency,encoder,assembler,lr1))
    val model1 = pipeline1.fit(training)

    // Make predictions on test data. model is the model with combination of parameters that performed best.
    model1.transform(training).select( "final_status", "probability","predictions").show()

    //predict on test data
    val predicResult = model1.transform(test)
    predicResult.select( "final_status", "probability","predictions").show()

    //calculate f1 score
    val f1evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val accuracy = f1evaluator.evaluate(predicResult)
    println("F1 score before model tuning " , accuracy, "****************************** " )
    //0.6146923592544336
    println(+ accuracy)

    //Grid of parameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr1.regParam, Array(0.000000001, 0.0000001, 0.00001, 0.001))
      .addGrid(tf.minDF, Array(55.0, 75.0, 95.0))
      .build()

    //Train the model, 70% of the data will be used for training and the remaining 30% for validation.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline1)
      .setEvaluator(f1evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // Run train validation split, and choose the best set of parameters.
    val bestModel = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters that performed best.
    val dfWithPredictions = bestModel.transform(test)

    //confusion matrix shows the comparation of the data and prediction result
    dfWithPredictions.groupBy("final_status", "predictions").count().show()

    //show the f1 accuracy
    val accuracyf1 = f1evaluator.evaluate(dfWithPredictions)
    println("F1 score after model tuning = " , accuracyf1, "************************")
    //=0.6546370572388954

    bestModel.write.overwrite().save("lrm_model.model")

    spark.stop()

  }


}
