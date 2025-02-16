import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, col, to_timestamp
from pyspark.ml.feature import VectorAssembler, StandardScaler


spark = SparkSession.builder.appName("COMP4107").master("local[*]").getOrCreate()


# - `VectorAssembler` merges feature into one vector: `[open, high, low, close, volume]`
#     - LSTMs expect input to be multi-dimensional as a tensor with shape: $$ \text{(batch\_size, sequence\_length, feature\_dimension)} $$
#     - `feature_dimension` corresponds to the number of features, 5 in this case
# - model will learn different weights for each dimension, allowing it to understand the relationship between features


def stop_spark():
    spark.stop()


def load_and_preprocess_data(filepath: str):
    # loading data into pyspark dataframe
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    # converting the UNIX timestamp to proper datetime format
    df = df.withColumn("datetime", to_timestamp(from_unixtime(col("Timestamp"))))
    df.drop("Timestamp")
    # sorting to maintain time order
    df = df.orderBy("datetime")
    # handling missing values
    df = df.na.fill(0)
    df = df.filter(col("datetime").isNotNull())

    # feature assembly and scaling
    # combining cols into one feature vector, needed for spark ML
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    # scaling features for zero mean and unit variance
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    # analyzes `features` in `df` and computes the std deviation and mean for each feature
    scalerModel = scaler.fit(df)
    # applying the scaling transformation to the data using the statistics from the last line
    df = scalerModel.transform(df)

    preprocessed_df = df.select("datetime", "scaledFeatures")
    return preprocessed_df
