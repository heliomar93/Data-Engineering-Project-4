import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql.types import TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['default']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['default']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Create a Spark session to extract, transform and load the data
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Process the song data, transform and load the song and artist table in the output bucket
    """
    
    # get filepath to song data file
    song_data = os.path.join(input_data + 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select('song_id', 'title', 'artist_id', 'year', 'duration').dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(os.path.join(output_data + 'songs_table'), partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artists_table = df.withColumnRenamed('artist_name', 'name') \
                      .withColumnRenamed('artist_location', 'location') \
                      .withColumnRenamed('artist_latitude', 'lattitude') \
                      .withColumnRenamed('artist_longitude', 'longitude')
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data + 'artists_table'), partitionBy=['name'])


def process_log_data(spark, input_data, output_data):
    """
    Process the log data and load the result on the output folder. Also join the log and song data to create the songplays_table.
    """

    # get filepath to log data file
    log_data = os.path.join(input_data + 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level')
    users_table = users_table.withColumnRenamed('userId', 'user_id') \
                             .withColumnRenamed('firstName', 'first_name') \
                             .withColumnRenamed('lastName', 'last_name')
    users_table = users_table.dropDuplicates(['user_id', 'level'])
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data + 'users_table'), partitionBy=['first_name'])
    
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x/1000)), TimestampType())
    df = df.withColumn("start_time", get_timestamp(df.ts))
    
    # extract columns to create time table
    time_table = df.select('start_time') \
                   .withColumn('hour', hour('start_time')) \
                   .withColumn('day', dayofmonth('start_time')) \
                   .withColumn('week', weekofyear('start_time')) \
                   .withColumn('month', month('start_time')) \
                   .withColumn('year', year('start_time')) \
                   .withColumn('weekday', dayofweek('start_time')) \
                   .dropDuplicates()
        
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(os.path.join(output_data + 'time_table'), partitionBy=['year', 'month'])

    # read in song data to use for songplays table
    song_df = os.path.join(input_data + 'song_data/*/*/*/*.json')

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name) & (df.length == song_df.duration), 'left_outer').select(
            df.start_time,
            col("userId").alias('user_id'),
            df.level,
            song_df.song_id,
            song_df.artist_id,
            col("sessionId").alias("session_id"),
            df.location,
            col("useragent").alias("user_agent"),
            year('datetime').alias('year'),
            month('datetime').alias('month'))

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(os.path.join(output_data + 'songplays_table'), partitionBy=['year', 'month'])

def main():
    """
    Run the ETL process, extracting, transforming and loading the data.
    """
    #Spark session
    spark = create_spark_session()
    
    #Input data
    input_data = "s3a://udacity-dend/"
    #output data
    output_data = "s3a://sparkfydataheliomar/"
    
    #Process song data
    process_song_data(spark, input_data, output_data)
    
    #Process log data
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
