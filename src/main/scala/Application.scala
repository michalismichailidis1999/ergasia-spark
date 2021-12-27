import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{max, min, col}

import scala.collection.mutable.ListBuffer
import scala.math.{pow, sqrt}

object Application {
  case class Point(x: Float, y: Float)

  def main(args: Array[String]): Unit = {
    // Check if user passed any arguments while running the program
    if (args.length == 0) {
      println("File path is required. Please pass one argument when you run the application.")
      return
    }

    // Time the application started
    val startTime = System.nanoTime

    // Path from the file which is taken from the arguments
    val filename = args(0)

    // Creating spark session
    val sparkSession = SparkSession.builder().appName("Ergasia Spark").master("local").getOrCreate()

    // Creating the dataframe schema which is going to have
    // 'X' and 'Y' as labels (dataframe columns)
    val schema = StructType(
      StructField("X", FloatType, true) :: StructField("Y", IntegerType, true) :: Nil
    )

    // Creating the dataframe
    val df = sparkSession
      .read
      .option("header", "false") // datafile isn't containing a header
      .schema(schema)
      .csv(filename)
      .na.drop() // drop all null values

    val normalizedDf = this.normalize(df) // normalize the data (range will be 0-1)

    // transform df with VectorAssembler to add feature column
    val cols = Array("X", "Y")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    // the featureDf will contain all df columns and also one new column 'features'
    // in which every row is going to be an array of 2 values x and y
    val featureDf = assembler.transform(normalizedDf)

    // Trains a k-means model with 5 clusters
    val kmeans = new KMeans().setK(5).setFeaturesCol("features").setPredictionCol("prediction")
    val model = kmeans.fit(featureDf)

    // predictDf will contain all featureDf columns + prediction column
    val predictDf = model.transform(featureDf)

    // find all outliers
    findOutliers(predictDf)

    // print total main execution time
    println(s"Program execution finished after ${(System.nanoTime - startTime) / 1e9d} seconds")
    println("")

    // stop spark session
    sparkSession.stop()
  }

  def findOutliers(df: DataFrame): Unit = {
    // Because we don't know how much data a cluster will contains
    // we initialize clusters as a list buffer
    // and in a for loop we are going to append depending on the cluster class of a point
    val clusters = ListBuffer[ListBuffer[Point]]()

    // add 5 list buffers because we used 5 clusters in our k means algorithm
    for(_ <- 0 until 5){
      clusters.append(ListBuffer[Point]())
    }

    df.collect().foreach(row => {
      // cluster class will be a number from 0 to 4 (because we have 5 clusters only)
      val cluster_class = row(3).asInstanceOf[Int]
      val x = row(0).asInstanceOf[Float]
      val y = row(1).asInstanceOf[Float]

      clusters(cluster_class).append(Point(x, y))
    })

    // we are going to compute k nearest neighbors so we need to specify the number of neighbors
    val numOfNeighbors = 10

    // initialize outliers as a list buffer because we don't know how many outliers our algorithm is going to find
    val outliers = ListBuffer[(Float, Float)]()

    // Compute Local Outlier Factor for every point in every cluster
    // using a different thread for each cluster
    val threads = for(i <- 0 until 5) yield new Thread{
      override def run: Unit = {
        val cluster = clusters(i)

        // get k nearest neighbors
        // our algorithm is work exactly like python, is returns distances and indices array
        val neighbors = KNearestNeighbors(cluster, numOfNeighbors)

        // distances and indices are an n x m array where n -> number of points in cluster
        // and m -> number of nearest neighbors
        // each row in distances array contains the top k closest distances from neighbor points
        // and the corresponding row in indices array the point index in the cluster
        val distances = neighbors._1
        val indices = neighbors._2

        // reachability distances
        // because the distances array is sorted we just need to get the last column from the whole array
        val rd = distances.map(distArr => distArr(distArr.length - 1))

        // initialize local reachability density array
        val lrd = Array.ofDim[Float](distances.length)

        // Compute local reachability density for every point
        // which is the inverse average from it's neighbors reachability distance
        distances.zip(0 until distances.length).foreach(tup1 => {
          val distArr = tup1._1
          val i = tup1._2

          var sum: Float = 0

          distArr.zip(0 until distArr.length).foreach(tup2 => {
            val dist = tup2._1
            val j = tup2._2

            // if the reachability distance is less than the distance between the points then add their distance
            // else add the reachability distance of the current point
            if (dist > rd(indices(i)(j))) {
              sum += dist
            } else {
              sum += rd(indices(i)(j))
            }
          })

          // now compute the lrd which is the inverse of the average of the reachability distance
          lrd(i) = 1 / (sum / numOfNeighbors)
        })

        // Initialize local outlier factor array
        val lof = Array.ofDim[Float](indices.length)

        // now compute local outlier factor of each point
        // the formula is the average of the lrd divided by the lrd of the current point we are computing the lof
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

          // if the local outlier factor is greater that means that the point is an outlier
          if (factor > 1) {
            outliers.append((cluster(index).x, cluster(index).y))
          }
        })
      }
    }

    // starts all threads
    threads.foreach(_.start())
    // wait for all the threads to finish before we continue
    threads.foreach(_.join())

    // Print outliers
    println("Outliers: ")
    outliers.foreach(point => println(s"${point._1},${point._2}"))
    println("----------------------------------")
  }

  def KNearestNeighbors(data: ListBuffer[Point], numOfNeighbors: Int): (Array[Array[Float]], Array[Array[Int]]) = {
    // initialize distances and indices arrays
    val distances = Array.ofDim[Float](data.size, numOfNeighbors)
    val indices = Array.ofDim[Int](data.size, numOfNeighbors)

    // compute top k nearest neighbors using brute force
    data.zip(0 until data.size).foreach(tup1 => {
      val p = tup1._1
      val index = tup1._2

      // store the distance between point p and point p_2 in index 0
      // and in index 1 the index of the point p_2
      val arr = Array.ofDim[(Float, Int)](data.size)

      data.zip(0 until data.size).foreach(tup2 => {
        val p_2 = tup2._1
        val index_2 = tup2._2

        arr(index_2) = (sqrt(pow(p.x - p_2.x, 2) + pow(p.y - p_2.y, 2)).asInstanceOf[Float], index_2)
      })

      // sort the array in ascending order, using the distances
      val sortedArr = arr.sortWith(_._1 < _._1)

      // get next 5 elements skipping the index 0 which is going to be 0 every time
      // since the distance between itself is always zero
      distances(index) = sortedArr.map(_._1).slice(1, numOfNeighbors)
      indices(index) = sortedArr.map(_._2).slice(1, numOfNeighbors)
    })

    // return distances and indices
    (distances, indices)
  }

  def normalize(df: DataFrame): DataFrame = {
    // get max from column X
    val maxX = df.agg(max("X")).collect()(0)(0).asInstanceOf[Float]
    // get min from column X
    val minX = df.agg(min("X")).collect()(0)(0).asInstanceOf[Float]
    // compute the denominator which is max - min
    val denX = maxX - minX

    // Same for Y
    val maxY = df.agg(max("Y")).collect()(0)(0).asInstanceOf[Int]
    val minY = df.agg(min("Y")).collect()(0)(0).asInstanceOf[Int]
    val denY = maxY - minY

    // change columns X and Y using the formula ((Xi - x_min) / (x_max - x_min))
    // and also change their type from double to float for less precision to make the performance better
    val normalizedDf = df.withColumn("X", (col("X") - minX) / denX)
      .withColumn("X", col("X").cast(FloatType))
      .withColumn("Y", (col("Y") - minY) / denY)
      .withColumn("Y", col("Y").cast(FloatType));

    normalizedDf
  }
}
