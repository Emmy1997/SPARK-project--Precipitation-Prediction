# import findspark
# findspark.init()
from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
import os
import pyspark.sql.functions as F
from pyspark.sql.types import *

def init_spark(app_name: str):
  spark = SparkSession.builder.appName(app_name).getOrCreate()
  sc = spark.sparkContext
  sc.setLogLevel('OFF')
  return spark, sc

def connect_write(df):
    # CONNECT TO SQL SERVER AND LOAD RESULTS
    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"

    table_name = "streamingWeather"
    username = "ilanit0sobol"
    password = "Qwerty12!"

    try:
        df.write \
            .format("jdbc") \
            .mode("append") \
            .option("url", url) \
            .option("dbtable", table_name) \
            .option("user", username) \
            .option("password", password) \
            .save()
    except ValueError as error:
        print("Connector write failed", error)

def connect_read(df):
    # CONNECT TO SQL SERVER AND LOAD RESULTS
    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"

    table_name = "streamingWeather"

    username = "ilanit0sobol"
    password = "Qwerty12!"

    try:
        df = df.read \
            .format("jdbc") \
            .mode("append") \
            .option("url", url) \
            .option("dbtable", table_name) \
            .option("user", username) \
            .option("password", password) \
            .load()
    except ValueError as error:
        print("Connector write failed", error)

    return df

def ETL():
    os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8," \
                                 "com.microsoft.azure:spark-mssql-connector_2.12:1.1.0 pyspark-shell"

    spark, sc = init_spark('demo')
    df_spark = spark.read.text('ghcnd-stations.txt')
    df_spark = df_spark.select(df_spark.value.substr(0, 11).alias('StationIdA'),
                      df_spark.value.substr(12, 9).cast("float").alias('LATITUDE'),
                      df_spark.value.substr(22, 9).cast("float").alias('LONGITUDE'),
                      df_spark.value.substr(32, 9).cast("float").alias('ELEVATION'))

    raw_stream_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", 'dds2020s-kafka.eastus.cloudapp.azure.com:9092') \
        .option("subscribe", "CH, GM, GR, BR, IS")\
        .option("maxOffsetsPerTrigger", 500000) \
        .option("startingOffsets", "earliest")\
        .load()
    # streaming contains multiple attributes, however the "data" is in 'value':
    string_value_df = raw_stream_df.selectExpr("CAST(value AS STRING)")
    # Define the schema of the data:
    noaa_schema = StructType([StructField('StationId', StringType(), False),
                              StructField('Date', IntegerType(), False),
                              StructField('Variable', StringType(), False),
                              StructField('Value', IntegerType(), False),
                              StructField('M_Flag', StringType(), True),
                              StructField('Q_Flag', StringType(), True),
                              StructField('S_Flag', StringType(), True),
                              StructField('ObsTime', StringType(), True)])

    # Read each 'value' String as JSON:
    json_df = string_value_df.select(F.from_json(F.col("value"), schema=noaa_schema).alias('json'))
    # Flatten the nested object:
    features = ['TMAX', 'TMIN', 'PRCP']
    cols =  ['StationId', 'Date', 'Variable', 'Value']
    streaming_df = json_df.select("json.*")
    #filter only relevant variables and years
    streaming_df = streaming_df.filter(F.col('Date') >= 19700101).\
        filter('Q_Flag is null')\
        .filter(F.col('Variable').isin(features)).select(cols)

    def foreachfunc(df, epoch_id):
        global batch_counter
        batch_counter = batch_counter + 1
        print('BATCH START NUM:'  + str(batch_counter))
        features = ['TMAX', 'TMIN', 'PRCP']
        cols = ['StationId',"Month", 'Year', 'SumTmax', 'SumTmin','SumPrcp',
                'countTmax','countTmin' ,'countPrcp', 'FIPS_code']
        ##APPLY TRANSFORMATION AND CLEAN DATA
        df = df \
            .groupby('StationId', 'Date') \
            .pivot('Variable') \
            .agg(F.first('Value')) \
          .withColumn("Month", ((F.col("Date")/100) % 100).cast(IntegerType()))\
          .withColumn('Year', (F.col("Date")/10000).cast(IntegerType()))
        #make sure missing columns are filled with nulls if didnt appear in batch - to fit table scheme
        for col in df.columns:
            if col in features:
                features.remove(col)
        for col in features:
            df = df.withColumn(col, lit(None).cast(StringType()))
        #apply aggregated computations
        df = df.groupby('StationId','Month', 'Year').\
            agg(F.sum(F.col('TMAX')).alias('SumTmax'), F.sum(F.col('TMIN')).alias('SumTmin'),
                 F.sum(F.col('PRCP')).alias('SumPrcp'),
                 F.approx_count_distinct(F.col('TMAX')).alias('countTmax'),
                 F.approx_count_distinct(F.col('TMIN')).alias('countTmin'),
                 F.approx_count_distinct(F.col('PRCP')).alias('countPrcp'))\
            .withColumn('FIPS_code', F.col("StationId").cast(StringType()).substr(0, 2))

        df = df.select(cols)
        #join with static spark dateframe - enrich data
        df = df.join(df_spark, df_spark.StationIdA == df.StationId, how='left_outer').drop('StationIdA').\
            withColumn('Batch_id', lit(epoch_id).cast(StringType()))
        df.show(5)
        connect_write(df)
        print('FINISHED BATCH')
        pass

    print('starting')
    query = streaming_df\
            .writeStream \
            .trigger(processingTime='2 seconds') \
            .foreachBatch(foreachfunc)\
            .outputMode('update').start() \

    print(query.isActive)
    query.awaitTermination()
    print('done')

def finishETL():
    """
    compute average and remove row if it has any null value
    :return:
    """
    os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1," \
                                        "com.microsoft.azure:spark-mssql-connector_2.12:1.1.0 pyspark-shell"
    spark, sc = init_spark('ETL')
    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"
    table_name = "streamingWeather"
    username = "ilanit0sobol"
    password = "Qwerty12!"
    df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("dbtable", table_name) \
        .option("user", username) \
        .option("password", password) \
        .load()

    cols = ['StationId', 'Month', 'Year', 'AvgTmax', 'AvgTmin', 'AvgPrcp', 'FIPS_code', 'LATITUDE', 'LONGITUDE',
            'ELEVATION']
    df = df.groupby('StationId', 'Month', 'Year', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'FIPS_code').\
        agg(((F.sum(F.col('SumPrcp')))/F.sum(F.col('countPrcp'))).alias('AvgPrcp'),
            ((F.sum(F.col('SumTmin')))/F.sum(F.col('countTmin'))).alias('AvgTmin'),
            ((F.sum(F.col('SumTmax'))) / F.sum(F.col('countTmax'))).alias('AvgTmax'))

    df = df.select(cols)
    df = df.filter("AvgTmin is not null").filter("AvgTmax is not null").filter("AvgPrcp is not null")

    server_name = "jdbc:sqlserver://technionddscourse.database.windows.net:1433"
    database_name = "ilanit0sobol"
    url = server_name + ";" + "databaseName=" + database_name + ";"

    table_name = "Weather"
    username = "ilanit0sobol"
    password = "Qwerty12!"

    try:
        df.write \
            .format("jdbc") \
            .mode("append") \
            .option("url", url) \
            .option("dbtable", table_name) \
            .option("user", username) \
            .option("password", password) \
            .save()
    except ValueError as error:
        print("Connector write failed", error)






if __name__ == '__main__':
    ETL()
    finishETL()