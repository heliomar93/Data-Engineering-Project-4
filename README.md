# Data Lake

## Introduction

A music streaming startup, Sparkify, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

This project extract the data from a S3 bucket, in JSON, processes this data, and load on another S3 bucket on parquet.

## Input data

### Song dataset
The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID.

The files are on the format belows:

~~~~
{"num_songs": 1, 
"artist_id": "ARJIE2Y1187B994AB7", 
"artist_latitude": null, 
"artist_longitude": null, 
"artist_location": "", 
"artist_name": "Line Renaud", 
"song_id": "SOUPIRU12A6D4FA1E1", 
"title": "Der Kleine Dompfaff", 
"duration": 152.92036, 
"year": 0}
~~~~

The files of song dataset are on the S3 bucket: 

```
s3://udacity-dend/song_data
```

### Logs dataset
The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The files with the logs are on the following format:

~~~~
{"artist":null,
"auth":"Logged In",
"firstName":"Lily",
"gender":"F",
"itemInSession":0,
"lastName":"Burns",
"length":null,
"level":"free",
"location":"New York-Newark-Jersey City, NY-NJ-PA",
"method":"GET",
"page":"Home",
"registration":1540621059796.0,
"sessionId":689,
"song":null,
"status":200,
"ts":1542592468796,
"userAgent":"\"Mozilla\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/36.0.1985.125 Safari\/537.36\"",
"userId":"32"}
~~~~

The files of logs dataset are on the S3 bucket: 

```
s3://udacity-dend/log_data
```
## Output data

### Songplays dataset
These files contains the songs played. To achieve only the musics played, the dataset is filtered by the column page to have only the NextPage records.
There are 9 columns here: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location and user_agent.
We get this dataset joining the logs and the songs dataset, by the columns that contains the name of the song, the name of the artist and the length of the song.
The files are partitioned by year and month.

### Users dataset
These tables contains the users. To reach the users, we select the columns from the logs data.
There are 5 columns here: user_id, first_name, last_name, gender and level.
In the output database, the infos are partitioned by the first name of the user.

### Songs dataset
These files contains songs in the music database.
We get this data from the song data. We select 5 columns: song_id, title, artist_id, year and duration.
The data in the output bucket are partitioned by the year of the song and the artist_id.

### Artists dataset
We have here the information about the artists.
We get this data select the columsn from the songs dataset.
There are 5 columsn here: artist_id, name, location, lattitude, longitude.
The data in the output bucket are partitioned by the name of the artist.

### Time dataset
We have here the files about the time of the songplays.
The data are stored in 7 columns:start_time, hour, day, week, month, year and weekday.
We get this data processing the data from the column start_time from the logs data.

## Files of the etl
This repository contains the etl file that extract, transform and load the data in the output bucket. To run this properly, you need to make the dl.cfg file, with the AWS secret access key and the access key ID.