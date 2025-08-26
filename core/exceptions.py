"""
Professional Exception Handling System
Custom exceptions for better error management and debugging.
"""

class ReliconError(Exception):
    """Base exception for all Relicon errors."""
    
    def __init__(self, message: str, error_code: str = None, context: dict = None):
        self.message = message
        self.error_code = error_code or "RELICON_ERROR"
        self.context = context or {}
        super().__init__(self.message)

class BrandAnalysisError(ReliconError):
    """Raised when brand analysis fails."""
    
    def __init__(self, message: str, brand_name: str = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="BRAND_ANALYSIS_ERROR",
            context={**(context or {}), "brand_name": brand_name}
        )

class LogoAnalysisError(ReliconError):
    """Raised when logo analysis fails."""
    
    def __init__(self, message: str, file_path: str = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="LOGO_ANALYSIS_ERROR", 
            context={**(context or {}), "file_path": file_path}
        )

class VideoGenerationError(ReliconError):
    """Raised when video generation fails."""
    
    def __init__(self, message: str, service_type: str = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="VIDEO_GENERATION_ERROR",
            context={**(context or {}), "service_type": service_type}
        )

class ValidationError(ReliconError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context={**(context or {}), "field": field}
        )

class ConfigurationError(ReliconError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context={**(context or {}), "config_key": config_key}
        )

class ServiceError(ReliconError):
    """Raised when external service fails."""
    
    def __init__(self, message: str, service: str = None, status_code: int = None, context: dict = None):
        super().__init__(
            message=message,
            error_code="SERVICE_ERROR",
            context={**(context or {}), "service": service, "status_code": status_code}
        )
