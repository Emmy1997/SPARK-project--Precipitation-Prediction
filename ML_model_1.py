import findspark
findspark.init()
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import numpy as np
import matplotlib.pyplot as plt


def init_spark(app_name: str):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def load_df():
    os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.4," \
                                        "com.microsoft.azure:spark-mssql-connector_2.12:1.1.0 pyspark-shell"
    spark, sc = init_spark('learning')
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
    return df


def cv():
    df = load_df()
    df = df.select("Month", "Year", "FIPS_code", "AvgTmin", "AvgTmax", "LONGITUDE", "LATITUDE", "ELEVATION", "AvgPrcp")
    indexer = StringIndexer(inputCol="FIPS_code", outputCol="FIPS_index")
    encoder = OneHotEncoder(inputCol="FIPS_index", outputCol="FIPS_vec")
    assembler = VectorAssembler(inputCols=["Month", "Year", "AvgTmin", "AvgTmax", "LONGITUDE", "LATITUDE", "ELEVATION",
                                           "FIPS_vec"], outputCol="features")
    lr = LinearRegression(labelCol="AvgPrcp", featuresCol="features")
    pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
    params = ParamGridBuilder() \
        .addGrid(lr.maxIter, [1, 5, 10]) \
        .addGrid(lr.regParam, [0.1, 0.5, 1]) \
        .addGrid(lr.elasticNetParam, [0, 0.5, 0.8]) \
        .build()
    evaluator = RegressionEvaluator() \
        .setMetricName("rmse") \
        .setPredictionCol("prediction") \
        .setLabelCol("AvgPrcp")
    # CV (Model Selection)
    CV = CrossValidator(collectSubModels=True) \
        .setEstimator(pipeline) \
        .setEvaluator(evaluator) \
        .setEstimatorParamMaps(params) \
        .setNumFolds(3)
    # Run cross-validation, and choose the best set of parameters.
    cvModel = CV.fit(df)
    print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])


def learning():
    df = load_df()
    df = df.select("Month", "Year", "FIPS_code", "AvgTmin", "AvgTmax", "LONGITUDE", "LATITUDE", "ELEVATION", "AvgPrcp")
    indexer = StringIndexer(inputCol="FIPS_code", outputCol="FIPS_index")
    encoder = OneHotEncoder(inputCol="FIPS_index", outputCol="FIPS_vec")
    assembler = VectorAssembler(inputCols=["Month", "Year", "AvgTmin", "AvgTmax", "LONGITUDE", "LATITUDE", "ELEVATION",
                                           "FIPS_vec"], outputCol="features")
    transformer_pipeline = Pipeline(stages=[indexer, encoder, assembler])
    model = transformer_pipeline.fit(df)
    prepared_df = model.transform(df)
    train, test = prepared_df.randomSplit([0.7, 0.3])

    # Model's parameters were chosen by cross-validation procedure
    lr = LinearRegression(labelCol="AvgPrcp", featuresCol="features").setMaxIter(1).setRegParam(1.0) \
        .setElasticNetParam(0.8)

    # Fit model on train data
    lrModel = lr.fit(train)

    # Make predictions on train and test data, evaluate the model by RMSE
    out_train = lrModel.transform(train) \
        .select("prediction", "AvgPrcp").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(out_train)
    print("RMSE train: " + str(metrics.rootMeanSquaredError))

    out_test = lrModel.transform(test) \
        .select("prediction", "AvgPrcp").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(out_test)
    print("RMSE test: " + str(metrics.rootMeanSquaredError))

    # Compute RMSE per country
    countries = ['CH', 'BR', 'GM', 'GR', 'IS']
    rmse_train = []
    rmse_test = []
    for country in countries:
        filtered_train = train.filter(F.col("FIPS_code") == country)
        prediction_train = lrModel.transform(filtered_train).select("prediction", "AvgPrcp") \
            .rdd.map(lambda x: (float(x[0]), float(x[1])))
        metrics = RegressionMetrics(prediction_train)
        rmse_train.append(metrics.rootMeanSquaredError)

        filtered_test = test.filter(F.col("FIPS_code") == country)
        prediction_test = lrModel.transform(filtered_test).select("prediction", "AvgPrcp") \
            .rdd.map(lambda x: (float(x[0]), float(x[1])))
        metrics = RegressionMetrics(prediction_test)
        rmse_test.append(metrics.rootMeanSquaredError)
    print("train: {}".format(rmse_train))
    print("test: {}".format(rmse_test))


def learning_without_temp():
    df = load_df()
    df = df.select("Month", "Year", "FIPS_code", "LONGITUDE", "LATITUDE", "ELEVATION", "AvgPrcp")
    indexer = StringIndexer(inputCol="FIPS_code", outputCol="FIPS_index")
    encoder = OneHotEncoder(inputCol="FIPS_index", outputCol="FIPS_vec")
    assembler = VectorAssembler(inputCols=["Month", "Year", "LONGITUDE", "LATITUDE", "ELEVATION", "FIPS_vec"],
                                outputCol="features")
    transformer_pipeline = Pipeline(stages=[indexer, encoder, assembler])
    model = transformer_pipeline.fit(df)
    prepared_df = model.transform(df)
    train, test = prepared_df.randomSplit([0.7, 0.3])

    # Model's parameters were chosen by cross-validation procedure
    lr = LinearRegression(labelCol="AvgPrcp", featuresCol="features").setMaxIter(1).setRegParam(1.0) \
        .setElasticNetParam(0.8)

    # Fit model on train data
    lrModel = lr.fit(train)

    # Make predictions on train and test data, evaluate the model by RMSE
    out_train = lrModel.transform(train) \
        .select("prediction", "AvgPrcp").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(out_train)
    print("RMSE train no temp: " + str(metrics.rootMeanSquaredError))

    out_test = lrModel.transform(test) \
        .select("prediction", "AvgPrcp").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = RegressionMetrics(out_test)
    print("RMSE test no temp: " + str(metrics.rootMeanSquaredError))

    # Compute RMSE per country
    countries = ['CH', 'BR', 'GM', 'GR', 'IS']
    rmse_train = []
    rmse_test = []
    for country in countries:
        filtered_train = train.filter(F.col("FIPS_code") == country)
        prediction_train = lrModel.transform(filtered_train).select("prediction", "AvgPrcp") \
            .rdd.map(lambda x: (float(x[0]), float(x[1])))
        metrics = RegressionMetrics(prediction_train)
        rmse_train.append(metrics.rootMeanSquaredError)

        filtered_test = test.filter(F.col("FIPS_code") == country)
        prediction_test = lrModel.transform(filtered_test).select("prediction", "AvgPrcp") \
            .rdd.map(lambda x: (float(x[0]), float(x[1])))
        metrics = RegressionMetrics(prediction_test)
        rmse_test.append(metrics.rootMeanSquaredError)
    print("train (no temp): {}".format(rmse_train))
    print("test (no temp): {}".format(rmse_test))


def plot_hist():
    rmse_train = [35.4915045512981, 73.60486324889924, 16.37422794405875, 31.882898966004706, 28.697178103669717]
    rmse_test = [34.745306986555846, 69.1328810558554, 16.372318809468492, 30.142195843915488, 27.189064553882794]
    countries = ['China', 'Brazil', 'Germany', 'Greece', 'Israel']
    x = np.arange(len(countries))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(x + width / 2, rmse_train, width, label='train')
    ax.bar(x - width / 2, rmse_test, width, label='test')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Countries')
    ax.set_title('RMSE per country (without using temperature features)')
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    cv()
    learning()
    learning_without_temp()
    plot_hist()
