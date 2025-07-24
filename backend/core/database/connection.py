"""
Relicon AI Ad Creator - Database Connection
Database connection management and session handling
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from ..settings import settings

# Create SQLAlchemy engine with optimized settings
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,          # Validate connections before use
    pool_recycle=300,            # Recycle connections every 5 minutes
    pool_size=10,                # Connection pool size
    max_overflow=20,             # Maximum overflow connections
    echo=settings.DEBUG,         # Log SQL queries in debug mode
    connect_args={
        "connect_timeout": 10,   # Connection timeout
        "application_name": "relicon_ai_ad_creator"
    } if "postgresql" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Keep objects accessible after commit
)

# Base class for all database models
Base = declarative_base()


def get_database() -> Session:
    """
    FastAPI dependency for database sessions
    
    Provides a database session for each request with automatic cleanup.
    Use this as a dependency in FastAPI route handlers.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        # Rollback on any exception
        db.rollback()
        raise
    finally:
        # Always close the session
        db.close()


def init_database() -> None:
    """
    Initialize database with all tables
    
    Creates all tables defined in the models.
    Safe to call multiple times - will only create missing tables.
    """
    print("🗄️ Initializing database...")
    
    try:
        # Import all models to ensure they're registered with Base
        from .models import AdJob, UploadedAsset, GeneratedAsset, AdAnalytics, SystemMetrics
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {str(e)}")
        raise


def test_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        # Try to execute a simple query
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"❌ Database connection test failed: {str(e)}")
        return False


def get_database_info() -> dict:
    """
    Get database connection information
    
    Returns:
        dict: Database connection details (without sensitive info)
    """
    try:
        url_parts = settings.DATABASE_URL.split('@')
        if len(url_parts) > 1:
            # Remove credentials from URL for logging
            host_db = url_parts[1]
            protocol = url_parts[0].split('://')[0]
            safe_url = f"{protocol}://[CREDENTIALS_HIDDEN]@{host_db}"
        else:
            safe_url = settings.DATABASE_URL
            
        return {
            "url": safe_url,
            "pool_size": engine.pool.size(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
            "checked_in": engine.pool.checkedin()
        }
    except Exception:
        return {"error": "Unable to retrieve database info"} 