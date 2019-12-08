package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.{udf,datediff,second,round,concat_ws,lower,from_unixtime}
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.log4j.PropertyConfigurator

object Preprocessor {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

      //Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
      //création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._
    PropertyConfigurator.configure("$HOME/spark-2.4.4-bin-hadoop2.7/conf/log4j.properties")

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("src/main/resources/train_clean.csv")

    df.show()

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)
    dfCasted.select("goal", "final_status").orderBy($"goal".desc).show(50)

    val df2: DataFrame = dfCasted.drop("disable_communication")
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    def cleanCountry(country:String, currency:String ):String={
      if (country=="False")
        currency
      else
        country
    }

    def cleanCurrency(currency:String):String ={
      if (currency != null && currency.length !=3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry : DataFrame = dfNoFutur.withColumn("country2",cleanCountryUdf($"country",$"currency"))
      .withColumn("currency2",cleanCurrencyUdf($"currency"))
      .drop("country","currency")

    val dfClean : DataFrame = dfCountry.withColumn("days_campaign",datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
        .withColumn("hours_prepa",round(second(from_unixtime($"launched_at"-$"created_at"))/3600,3))
        .drop("launched_at","created_at","deadline")
        .withColumn("text",lower(concat_ws(" ",$"name",$"desc",$"keywords")))
        .na.fill(Map("days_campaign" -> -1, "hours_prepa" -> -1, "goal"  -> -1 , "country2" -> "unkown" , "currency2" -> "unknown" ))

    dfClean.show()

    dfClean.write.mode("overwrite").parquet("src/main/resources/preprocessed_yang")


  }
}
