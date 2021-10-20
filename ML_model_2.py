import findspark
from pyspark.ml import Pipeline
findspark.init()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import os
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString, VectorIndexer
from pyspark.ml.classification import RandomForestClassifier


def init_spark(app_name: str):
  spark = SparkSession.builder.appName(app_name).getOrCreate()
  sc = spark.sparkContext
  sc.setLogLevel('OFF')
  return spark, sc

def split_df(df):
    """split df to 5 df by country"""

    cols = ['StationId', 'AvgPrcp', 'AvgTmin', 'AvgTmax', 'Year', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'Month']
    df = df.select(cols)
    df = df.withColumn('FIPS_code', F.col("StationId").cast(StringType()).substr(0, 2))
    df_china, df_germany, df_greece, df_brazil, df_israel = df.filter(df.FIPS_code == 'CH'), \
                                                            df.filter(df.FIPS_code == 'GM'), \
                                                            df.filter(df.FIPS_code == 'GR'), \
                                                            df.filter(df.FIPS_code == 'BR'), \
                                                            df.filter(df.FIPS_code == 'IS')

    return df_china, df_germany, df_greece, df_brazil, df_israel

def model_runner2(seasons_dict, df):
    """ run Random Forest with Season as predicted label.
    season_dict has season matching to months
    """
    features = ['AvgPrcp', 'AvgTmin', 'AvgTmax', 'Year', 'Location', 'Season']

    df = df.withColumn('Season', F.coalesce(*[F.when(F.col('Month').isin(months), int(season))
                                              for (season, months) in seasons_dict.items()]))

    df.show(5)
    labelIndexerMonth = StringIndexer(inputCol="Season", outputCol="indexedLabel").fit(df)
    locationAssembler = VectorAssembler(inputCols=['LATITUDE', 'LONGITUDE', 'ELEVATION'], outputCol='Location')
    featuresAssembler = VectorAssembler(inputCols=features, outputCol="features")
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)

    (train, test) = df.randomSplit([0.7, 0.3])
    # Train a RandomForest model
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexerMonth.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexerMonth, locationAssembler, featuresAssembler,
                                featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select example rows to display.
    predictions.select("predictedLabel", "Season", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    test_error = 1.0 - accuracy
    print("Test Error = %g" % (1.0 - accuracy))
    return test_error

def model_runner1(df):
    """
    run Random Forest with Month as label
    :param df: dataframe of current country
    :return:
    """
    features = ['AvgPrcp', 'AvgTmin', 'AvgTmax', 'Year', 'Location', 'Month']

    labelIndexerMonth = StringIndexer(inputCol="Month", outputCol="indexedLabel").fit(df)
    locationAssembler = VectorAssembler(inputCols=['LATITUDE', 'LONGITUDE', 'ELEVATION'], outputCol='Location')
    featuresAssembler = VectorAssembler(inputCols=features, outputCol="features")
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4)

    (train, test) = df.randomSplit([0.7, 0.3])
    # Train a RandomForest model
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexerMonth.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexerMonth, locationAssembler, featuresAssembler,
                                featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select example rows to display.
    predictions.select("predictedLabel", "Month", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    test_error = 1.0 - accuracy
    print("Test Error = %g" % (1.0 - accuracy))

    return test_error

def plot_errors(errors, name):
    fig, axs = plt.subplots()
    axs.set_title('Test Error by Country ' + name)
    axs.set_xlabel('Country')
    axs.set_ylabel('Error')
    width = 0.3
    labels = list(errors.keys())
    plot = axs.bar(range(5), list(errors.values()), width=0.3)
    axs.set_xticks(range(5))
    axs.set_xticklabels(labels)
    plt.legend()
    plt.show()

def Bonus_ML_RandomTree2():
    """run model on countries, predict Season
    """
    os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1," \
                                        "com.microsoft.azure:spark-mssql-connector_2.12:1.1.0 pyspark-shell"

    spark, sc = init_spark('demo')
    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"
    table_name = "Weather"
    username = "ilanit0sobol"
    password = "Qwerty12!"
    df = spark.read \
    .format("jdbc") \
    .option("url", url) \
    .option("dbtable", table_name) \
        .option("user", username) \
        .option("password", password) \
        .load()

    df_china, df_germany, df_greece, df_brazil, df_israel = split_df(df)
    season_encoder = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
    df_list = {'CH': df_china, 'GM': df_germany, 'GR': df_greece, 'BR': df_brazil, 'IS': df_israel}
    seasons_dict = {'CH': {'1': [3, 4], '2': [5, 6, 7], '3': [8, 9, 10], '4': [11, 12, 1, 2]},
                    'GM': {'1': [3, 4, 5], '2': [6, 7, 8], '3': [9, 10, 11], '4': [12, 1, 2]},
                    'GR': {'1': [3, 4, 5], '2': [6, 7, 8], '3': [9, 10, 11], '4': [12, 1, 2]},
                    'BR': {'1': [10, 11, 12], '2': [1, 2, 3], '3': [4, 5, 6], '4': [7, 8, 9]},
                    'IS': {'1': [3, 4, 5], '2': [6, 7, 8], '3': [9, 10, 11], '4': [12, 1, 2]}}
    errors = {'CH': 0, 'GM': 0, 'GR': 0, 'BR': 0, 'IS': 0}
    for country, df in df_list.items():
            errors[country] = model_runner2(seasons_dict[country], df)

    plot_errors(errors, 'ML model by Season')

def Bonus_ML_RandomTree1():
    """
    Run model on countries, predict Month and plot errors
    """
    os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1," \
                                        "com.microsoft.azure:spark-mssql-connector_2.12:1.1.0 pyspark-shell"

    spark, sc = init_spark('demo')
    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"
    table_name = "Weather"
    username = "ilanit0sobol"
    password = "Qwerty12!"
    df = spark.read \
    .format("jdbc") \
    .option("url", url) \
    .option("dbtable", table_name) \
        .option("user", username) \
        .option("password", password) \
        .load()

    df_china, df_germany, df_greece, df_brazil, df_israel = split_df(df)
    df_list = {'CH': df_china, 'GM': df_germany, 'GR':df_greece, 'BR':df_brazil, 'IS': df_israel}
    errors = {'CH': 0, 'GM': 0, 'GR': 0, 'BR': 0, 'IS': 0}
    for country, df in df_list.items():
        errors[country] = model_runner1(df)

    plot_errors(errors, 'ML model by month')



if __name__ == '__main__':
    Bonus_ML_RandomTree1()
    Bonus_ML_RandomTree2()