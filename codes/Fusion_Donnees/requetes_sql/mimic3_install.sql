-- Set mimic3 database path 
--using / instead of \

\set mimic_data_dir 'G:/mimic'

-- Drop database if mimic EXISTS
DROP DATABASE IF EXISTS mimic;

-- Create mimic DATABASE
CREATE DATABASE mimic OWNER postgres;

-- connect to mimic DATABASE
\c mimic;

--Create schema mimiciii

CREATE SCHEMA mimiciii;

--set default path to mimiciii
set search_path to mimiciii;

-- Create TABLES
\i c:/sql/postgres_create_tables.sql

-- Set the script to stop on errors

\set ON_ERROR_STOP 1

-- Load data into postgres DATABASE
\cd :mimic_data_dir
\i c:/sql/postgres_load_data.sql

-- Build INDEXES
\i c:/sql/postgres_add_indexes.sql

SELECT
	icustay_id, intime, outtime
from icustays
limit 10;

\i c:/sql/postgres_checks.sql