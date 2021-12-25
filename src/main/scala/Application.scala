import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{max, min, udf, col}

import scala.collection.mutable.ListBuffer
import scala.math.{pow, sqrt}

object Application {
  case class Point(x: Float, y: Float)

  def main(args: Array[String]): Unit = {
    //    if (args.length == 0) {
    //      println("File path is required. Please pass one argument when you run the application.")
    //      return
    //    }

    val startTime = System.nanoTime

    //    val filename = args(0)
    val filename = "./src/main/resources/data.csv"

    val sparkSession = SparkSession.builder().appName("Ergasia Spark").master("local").getOrCreate()

    val schema = StructType(
      StructField("X", FloatType, true) :: StructField("Y", IntegerType, true) :: Nil
    )

    val df = sparkSession
      .read
      .option("header", "false")
      .schema(schema)
      .csv(filename)
      .na.drop()

    val normalizedDf = this.normalize(df)

//    normalizedDf.collect().take(10).foreach(println)

    // transform df with VectorAssembler to add feature column
    val cols = Array("X", "Y")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(normalizedDf)

    // Trains a k-means model.
    val kmeans = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
    val model = kmeans.fit(featureDf)

    val predictDf = model.transform(featureDf)

    removeOutliers(predictDf)

    println(s"Program execution finished after ${(System.nanoTime - startTime) / 1e9d} seconds")
    println("")

    sparkSession.stop()
  }

  def removeOutliers(df: DataFrame): Unit = {
    var clusters = Vector(
      Vector[Point](),
      Vector[Point](),
      Vector[Point](),
      Vector[Point](),
      Vector[Point]()
    )

    df.collect().foreach(row => {
      val cluster_class = row(3).asInstanceOf[Int]
      val x = row(0).asInstanceOf[Float]
      val y = row(1).asInstanceOf[Float]

      val point = Point(x, y)
      val cluster = clusters(cluster_class) :+ point
      val updatedClusters = clusters.updated(cluster_class, cluster)

      clusters = updatedClusters
    })

    val numOfNeighbors = 10

    val outliers = ListBuffer[(Float, Float)]()

    val threads = for(i <- 0 until 5) yield new Thread{
      override def run: Unit = {
        val cluster = clusters(i)

        val neighbors = KNearestNeighbors(cluster, numOfNeighbors)

        val distances = neighbors._1
        val indices = neighbors._2

        val rd = distances.map(distArr => distArr(distArr.length - 1))

        val lrd = Array.ofDim[Float](distances.length)

        distances.zip(0 until distances.length).foreach(tup1 => {
          val distArr = tup1._1
          val i = tup1._2

          var sum: Float = 0

          distArr.zip(0 until distArr.length).foreach(tup2 => {
            val dist = tup2._1
            val j = tup2._2

            if (dist > rd(indices(i)(j))) {
              sum += dist
            } else {
              sum += rd(indices(i)(j))
            }
          })

          lrd(i) = 1 / (sum / numOfNeighbors)
        })

        val lof = Array.ofDim[Float](indices.length)

        indices.zip(0 until indices.length).foreach(tup1 => {
          val neighborsIndices = tup1._1
          val index = tup1._2

          var sum: Float = 0

          neighborsIndices.foreach(i => {
            sum += lrd(i)
          })

          lof(index) = (sum / numOfNeighbors) / lrd(index)
        })

        lof.zip(0 until lof.length).foreach(tup => {
          val factor = tup._1
          val index = tup._2

          if (factor > 1) {
            outliers.append((cluster(index).x, cluster(index).y))
          }
        })
      }
    }

    threads.foreach(_.start())
    threads.foreach(_.join())

    println("Non Outliers: ")
    outliers.foreach(point => println(s"${point._1},${point._2}"))
    println("----------------------------------")
  }

  def KNearestNeighbors(data: Vector[Point], numOfNeighbors: Int): (Array[Array[Float]], Array[Array[Int]]) = {
    val distances = Array.ofDim[Float](data.size, numOfNeighbors)
    val indices = Array.ofDim[Int](data.size, numOfNeighbors)

    data.zip(0 until data.size).foreach(tup1 => {
      val p = tup1._1
      val index = tup1._2

      val arr = Array.ofDim[(Float, Int)](data.size)

      data.zip(0 until data.size).foreach(tup2 => {
        val p_2 = tup2._1
        val index_2 = tup2._2

        arr(index_2) = (sqrt(pow(p.x - p_2.x, 2) + pow(p.y - p_2.y, 2)).asInstanceOf[Float], index_2)
      })

      val sortedArr = arr.sortWith(_._1 < _._1)

      distances(index) = sortedArr.map(_._1).slice(1, numOfNeighbors)
      indices(index) = sortedArr.map(_._2).slice(1, numOfNeighbors)
    })

    (distances, indices)
  }

  def normalize(df: DataFrame): DataFrame = {
    val maxX = df.agg(max("X")).collect()(0)(0).asInstanceOf[Float]
    val minX = df.agg(min("X")).collect()(0)(0).asInstanceOf[Float]
    val denX = maxX - minX

    val maxY = df.agg(max("Y")).collect()(0)(0).asInstanceOf[Int]
    val minY = df.agg(min("Y")).collect()(0)(0).asInstanceOf[Int]
    val denY = maxY - minY

    val normalizedDf = df.withColumn("X", (col("X") - minX) / denX)
      .withColumn("X", col("X").cast(FloatType))
      .withColumn("Y", (col("Y") - minY) / denY)
      .withColumn("Y", col("Y").cast(FloatType));

    normalizedDf
  }
}
