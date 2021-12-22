import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

import math.{pow, sqrt}
import scala.collection.mutable.ListBuffer

object Application {
  case class Point(x: Float, y: Int)

  def main(args: Array[String]): Unit = {
    if(args.length == 0){
      println("File path is required. Please pass one argument when you run the application.")
      return;
    }

    val startTime = System.nanoTime

    val filename = args(0)

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

    // transform df with VectorAssembler to add feature column
    val cols = Array("X", "Y")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(df)

    // Trains a k-means model.
    val kmeans = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
    val model = kmeans.fit(featureDf)

    val predictDf = model.transform(featureDf)

    removeOutliers(predictDf)

    println(s"Program execution finished after ${(System.nanoTime - startTime) / 1e9d} seconds")
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
      val y = row(1).asInstanceOf[Int]

      val point = Point(x, y)
      val cluster = clusters(cluster_class) :+ point
      val updatedClusters = clusters.updated(cluster_class, cluster)

      clusters = updatedClusters
    })

    val numOfNeighbors = 10

    val outliers = ListBuffer[(Float, Int)]()

    clusters.foreach(cluster => {
      val neighbors = KNearestNeighbors(cluster, numOfNeighbors)

      val distances = neighbors._1
      val indices = neighbors._2

      val rd = distances.map(distArr => distArr(distArr.size - 1))

      val lrd = Array.ofDim[Float](distances.size)

      distances.zip(0 until distances.size).foreach(tup1 => {
        val distArr = tup1._1
        val i = tup1._2

        var sum: Float = 0

        distArr.zip(0 until distArr.size).foreach(tup2 => {
          val dist = tup2._1
          val j = tup2._2

          if(dist > rd(indices(i)(j))){
            sum += dist
          }else{
            sum += rd(indices(i)(j))
          }
        })

        lrd(i) = 1 / (sum / numOfNeighbors)
      })

      val lof = Array.ofDim[Float](indices.size)

      indices.zip(0 until indices.size).foreach(tup1 => {
        val neighborsIndices = tup1._1
        val index = tup1._2

        var sum: Float = 0

        neighborsIndices.foreach(i => {
          sum += lrd(i)
        })

        lof(index) = (sum / numOfNeighbors) / lrd(index)
      })

      lof.zip(0 until lof.size).foreach(tup => {
        val factor = tup._1
        val index = tup._2

        if(factor > 1){
          outliers.append((cluster(index).x, cluster(index).y))
        }
      })
    })

    println("Outliers: ")
    outliers.foreach(point => println(s"${point._1},${point._2}"))
    print("----------------------------------")
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

        arr(index_2) = (sqrt(pow((p.x - p_2.x), 2) + pow((p.y - p_2.y), 2)).asInstanceOf[Float], index_2)
      })

      val sortedArr = arr.sortWith(_._1 < _._1)

      distances(index) = sortedArr.map(_._1).slice(1, numOfNeighbors)
      indices(index) = sortedArr.map(_._2).slice(1, numOfNeighbors)
    })

    (distances, indices)
  }
}
