"""
Version information for Relicon AI Video Generator
"""

__version__ = "0.6.4"
__version_info__ = (0, 6, 4)

# Version components
MAJOR = 0
MINOR = 6
PATCH = 4

# Build metadata
BUILD_NUMBER = None
RELEASE_TYPE = "stable"  # alpha, beta, rc, stable

def get_version():
    """Get formatted version string"""
    return __version__

def get_version_info():
    """Get version tuple"""
    return __version_info__

def get_full_version():
    """Get version with build info if available"""
    version = __version__
    if BUILD_NUMBER:
        version += f"+{BUILD_NUMBER}"
    return version
