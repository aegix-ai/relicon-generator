"""
Relicon AI Ad Creator - Database Layer (Legacy Import)
This file now imports from the modular database package for backward compatibility
"""

# Import all components from the new modular database package
from .database import *

# This file is kept for backward compatibility
# All database components are now organized in the core/database/ package:
# - connection.py: Database connection and session management
# - models.py: SQLAlchemy ORM models
# - repositories.py: Repository pattern implementations
# - manager.py: High-level database manager 