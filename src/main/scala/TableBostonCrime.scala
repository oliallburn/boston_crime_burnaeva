import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import PercentileApprox._
import org.apache.spark.sql.functions.collect_list


object TableBostonCrime extends App {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local")
    .appName("allburn")
    .getOrCreate()

  import spark.implicits._

  val crime = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(args(0))

  val offense_codes = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(args(1))

  offense_codes.cache()
  crime.cache()

  val code = offense_codes
    .withColumn("NAME_TYPE", split(col("NAME")," - ")
    .getItem(0))
    .drop("NAME")
      .orderBy("CODE")
      .dropDuplicates("CODE")

  val total_table = crime
      .join(broadcast(code), crime("OFFENSE_CODE") === code("CODE"))
      .drop("CODE")

  val crime_stat =  total_table
      .groupBy("DISTRICT")
      .agg(
        count("INCIDENT_NUMBER").as("crimes_total"),
        avg("Lat").as("lat"),
        avg("Long").as("lng")
      )
  val crime_per_month =  total_table
    .groupBy("DISTRICT", "YEAR", "MONTH")
    .agg(count("INCIDENT_NUMBER").as("crimes_total"))
      .orderBy("DISTRICT", "YEAR", "MONTH")

  val median_crime_sc = crime_per_month
    .groupBy("DISTRICT")
    .agg(percentile_approx($"crimes_total", lit(0.5))
    .as("crimes_monthly"))

  val partitionWindow = Window.partitionBy($"DISTRICT").orderBy($"count".desc)
  val rankTest = rank().over(partitionWindow)

  val frequent_crime = total_table
    .select("DISTRICT", "INCIDENT_NUMBER", "NAME_TYPE")
    .groupBy("DISTRICT", "NAME_TYPE")
    .count()
    .withColumn("rank", rankTest as "rank" )
    .filter($"rank" < 4)
    .groupBy("DISTRICT")
    .agg(concat_ws(", ", collect_list('NAME_TYPE)).as("frequent_crime_types"))

  val boston_crime = crime_stat
    .join(median_crime_sc, "DISTRICT")
      .join(frequent_crime, "DISTRICT" )
      .select($"DISTRICT", $"crimes_total", $"crimes_monthly", $"frequent_crime_types", $"lat", $"lng")
      .withColumnRenamed("DISTRICT", "district")

  boston_crime
    .repartition(1)
    .write
    .mode(SaveMode.Overwrite)
    .parquet(args(2))
}

