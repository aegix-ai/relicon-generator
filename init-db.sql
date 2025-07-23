-- Relicon AI Ad Creator - Database Initialization
-- Initialize the database with proper extensions and settings

-- Enable UUID extension for generating unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_stat_statements for performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Set timezone
SET timezone = 'UTC';

-- Create database (if not exists via environment)
-- The main database 'relicon' is created by POSTGRES_DB environment variable

-- Grant permissions to the postgres user
GRANT ALL PRIVILEGES ON DATABASE relicon TO postgres;

-- Create additional indexes for performance (will be created by SQLAlchemy models)
-- This is just a placeholder for any additional database setup

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres;

-- Log successful initialization
SELECT 'Relicon AI database initialized successfully!' as status; 