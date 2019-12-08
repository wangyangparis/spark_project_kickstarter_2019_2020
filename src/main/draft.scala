/*
package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.{udf,datediff,second,round,concat_ws,lower,from_unixtime}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.StringIndexer
class withoutpipeline {

  def main(args: Array[String]): Unit = {

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

    println("hello world ! from Trainer")


    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet("/Users/yang/coursMacbook/prepared_trainingset/")

    println("###################################",df)
    df.select("text").show()


    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val token=tokenizer.transform(df)

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val stopwords = remover.transform(token)

    val hashingTF = new HashingTF()
      .setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(stopwords)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    df.select("text").show()
    token.select("tokens").show()
    stopwords.select("filtered").show()
    rescaledData.select("features").show()

    val indexer_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexed_country = indexer_country.fit(df).transform(df)
    indexed_country.show()

    val indexer_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val indexed_currency = indexer_currency.fit(indexed_country).transform(indexed_country)
    indexed_currency.show()

    import org.apache.spark.ml.feature.OneHotEncoderEstimator


    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_1hot", "currency_1hot"))
    val model = encoder.fit(indexed_currency)

    val encoded = model.transform(indexed_currency)
    encoded.show()


    /*
        val tokenizer = new Tokenizer()
          .setInputCol("text")
          .setOutputCol("words")

        val hashingTF = new HashingTF()
          .setNumFeatures(1000)
          .setInputCol(tokenizer.getOutputCol)
          .setOutputCol("features")
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
        val

        val pipeline = new Pipeline()
          .setStages(Array(tokenizer, hashingTF, lr))
      */


  }



}
**/