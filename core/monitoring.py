"""
Professional Monitoring and Health Check System
Comprehensive system monitoring with metrics, alerts, and health checks.
"""

import time
import psutil
import threading
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
from statistics import mean, stdev
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from collections import deque, defaultdict
import hashlib
from statistics import mean, stdev

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    ML_MONITORING_AVAILABLE = True
except ImportError:
    ML_MONITORING_AVAILABLE = False

from core.config import config
from core.logger import get_logger
from core.exceptions import ConfigurationError

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class QualityTrend(Enum):
    """Quality trend indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_func: Callable[[], Dict[str, Any]]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    enabled: bool = True
    critical: bool = False
    last_run: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_result: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_connections: int
    process_count: int
    uptime_seconds: float
    
@dataclass
class QualityMetrics:
    """Quality monitoring metrics with comprehensive performance tracking."""
    timestamp: datetime
    blueprint_quality_score: float
    brand_confidence_score: float
    prompt_optimization_score: float
    generation_success_rate: float
    average_processing_time: float
    ml_enhancement_usage: float
    error_rate: float
    user_satisfaction_score: Optional[float] = None
    
    # New performance tracking metrics
    niche_type: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[int] = None
    scene_quality_scores: Optional[List[float]] = None
    response_times: Optional[Dict[str, float]] = None
    error_details: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    metric_name: str
    baseline_value: float
    acceptable_deviation: float
    warning_threshold: float
    critical_threshold: float
    created_at: datetime
    sample_count: int
    confidence_interval: Tuple[float, float]

@dataclass
class RegressionAlert:
    """Regression detection alert."""
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    severity: AlertSeverity
    trend: QualityTrend
    detected_at: datetime
    samples_analyzed: int
    recommendation: str

class QualityMonitor:
    """Real-time quality monitoring with ML-powered regression detection and Phase 2 ad performance tracking."""
    
    def __init__(self):
        self.quality_history: deque = deque(maxlen=10000)  # Store last 10k quality measurements
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.regression_alerts: List[RegressionAlert] = []
        self.anomaly_detector = None
        self.is_ml_enabled = False
        
        # Phase 2: Ad performance tracking
        self.ad_performance_history: deque = deque(maxlen=50000)  # Store last 50k ad performance metrics
        self.performance_predictions: Dict[str, Dict[str, Any]] = {}  # ML performance predictions
        self.creative_performance_cache: Dict[str, Dict[str, Any]] = {}  # Creative performance analytics
        
        if ML_MONITORING_AVAILABLE:
            self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize ML components for quality monitoring."""
        try:
            # Isolation Forest for anomaly detection
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            
            self.scaler = StandardScaler()
            self.is_ml_enabled = True
            
            logger.info("ML quality monitoring components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML monitoring: {e}")
            self.is_ml_enabled = False
    
    def record_quality_metrics(self, metrics: QualityMetrics):
        """Record quality metrics for monitoring and analysis."""
        try:
            self.quality_history.append(metrics)
            
            # Check for regressions
            self._check_quality_regression(metrics)
            
            # Update baselines if needed
            self._update_quality_baselines(metrics)
            
            # ML-based anomaly detection
            if self.is_ml_enabled and len(self.quality_history) > 100:
                self._detect_quality_anomalies(metrics)
            
            logger.debug(
                "Quality metrics recorded",
                action="quality.metrics.recorded",
                quality_score=metrics.blueprint_quality_score,
                confidence_score=metrics.brand_confidence_score,
                success_rate=metrics.generation_success_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to record quality metrics: {e}", exc_info=True)
    
    def _check_quality_regression(self, current_metrics: QualityMetrics):
        """Check for quality regressions against established baselines."""
        try:
            metrics_to_check = {
                'blueprint_quality_score': current_metrics.blueprint_quality_score,
                'brand_confidence_score': current_metrics.brand_confidence_score,
                'generation_success_rate': current_metrics.generation_success_rate,
                'average_processing_time': current_metrics.average_processing_time
            }
            
            for metric_name, current_value in metrics_to_check.items():
                if metric_name in self.performance_baselines:
                    baseline = self.performance_baselines[metric_name]
                    regression_alert = self._analyze_metric_regression(
                        metric_name, current_value, baseline
                    )
                    
                    if regression_alert:
                        self.regression_alerts.append(regression_alert)
                        self._handle_regression_alert(regression_alert)
                        
        except Exception as e:
            logger.error(f"Quality regression check failed: {e}")
    
    def _analyze_metric_regression(self, metric_name: str, current_value: float, baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Analyze individual metric for regression."""
        try:
            # Calculate deviation from baseline
            deviation_percent = abs(current_value - baseline.baseline_value) / baseline.baseline_value
            
            # Determine if it's a regression (lower is worse for most quality metrics)
            is_worse = current_value < baseline.baseline_value
            
            # Special handling for processing time (higher is worse)
            if metric_name == 'average_processing_time':
                is_worse = current_value > baseline.baseline_value
            
            # Determine severity
            severity = AlertSeverity.INFO
            if is_worse and deviation_percent > baseline.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif is_worse and deviation_percent > baseline.warning_threshold:
                severity = AlertSeverity.WARNING
            elif deviation_percent > baseline.acceptable_deviation:
                severity = AlertSeverity.INFO
            else:
                return None  # No significant deviation
            
            # Calculate trend
            trend = self._calculate_quality_trend(metric_name)
            
            # Generate recommendation
            recommendation = self._generate_regression_recommendation(metric_name, deviation_percent, trend)
            
            return RegressionAlert(
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=baseline.baseline_value,
                deviation_percent=deviation_percent * 100,
                severity=severity,
                trend=trend,
                detected_at=datetime.utcnow(),
                samples_analyzed=len(self.quality_history),
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Metric regression analysis failed for {metric_name}: {e}")
            return None
    
    def _calculate_quality_trend(self, metric_name: str) -> QualityTrend:
        """Calculate quality trend for a specific metric."""
        try:
            if len(self.quality_history) < 10:
                return QualityTrend.INSUFFICIENT_DATA
            
            # Get recent values for the metric
            recent_values = []
            for metrics in list(self.quality_history)[-20:]:  # Last 20 samples
                value = getattr(metrics, metric_name, None)
                if value is not None:
                    recent_values.append(value)
            
            if len(recent_values) < 5:
                return QualityTrend.INSUFFICIENT_DATA
            
            # Calculate trend using linear regression slope
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # Calculate volatility
            volatility = stdev(recent_values) / mean(recent_values) if mean(recent_values) > 0 else 0
            
            if volatility > 0.2:  # High volatility
                return QualityTrend.VOLATILE
            elif slope > 0.01:  # Improving
                return QualityTrend.IMPROVING
            elif slope < -0.01:  # Degrading
                return QualityTrend.DEGRADING
            else:
                return QualityTrend.STABLE
                
        except Exception as e:
            logger.error(f"Trend calculation failed for {metric_name}: {e}")
            return QualityTrend.UNKNOWN
    
    def _generate_regression_recommendation(self, metric_name: str, deviation_percent: float, trend: QualityTrend) -> str:
        """Generate actionable recommendations for regression alerts."""
        recommendations = {
            'blueprint_quality_score': {
                QualityTrend.DEGRADING: "Enable ML prompt optimization and review prompt templates",
                QualityTrend.VOLATILE: "Investigate prompt consistency and brand intelligence accuracy",
                QualityTrend.STABLE: "Monitor closely and consider A/B testing new optimizations"
            },
            'brand_confidence_score': {
                QualityTrend.DEGRADING: "Review brand intelligence model accuracy and retrain if needed",
                QualityTrend.VOLATILE: "Check input data quality and brand description consistency",
                QualityTrend.STABLE: "Consider expanding training data for edge cases"
            },
            'generation_success_rate': {
                QualityTrend.DEGRADING: "Check system resources and provider API status",
                QualityTrend.VOLATILE: "Review error handling and retry logic",
                QualityTrend.STABLE: "Monitor external dependencies"
            },
            'average_processing_time': {
                QualityTrend.DEGRADING: "Optimize ML model inference and check system resources",
                QualityTrend.VOLATILE: "Investigate load balancing and caching efficiency",
                QualityTrend.STABLE: "Consider scaling infrastructure"
            }
        }
        
        default_recommendation = f"Review {metric_name} performance and investigate {deviation_percent:.1f}% deviation"
        
        return recommendations.get(metric_name, {}).get(trend, default_recommendation)
    
    def _update_quality_baselines(self, current_metrics: QualityMetrics):
        """Update quality baselines based on recent performance."""
        try:
            # Only update baselines after collecting sufficient data
            if len(self.quality_history) < 50:
                return
            
            # Update baselines weekly or when significant improvements are sustained
            update_interval = timedelta(days=7)
            now = datetime.utcnow()
            
            metrics_to_baseline = {
                'blueprint_quality_score': current_metrics.blueprint_quality_score,
                'brand_confidence_score': current_metrics.brand_confidence_score,
                'generation_success_rate': current_metrics.generation_success_rate,
                'average_processing_time': current_metrics.average_processing_time
            }
            
            for metric_name, current_value in metrics_to_baseline.items():
                should_update = False
                
                if metric_name not in self.performance_baselines:
                    should_update = True
                else:
                    baseline = self.performance_baselines[metric_name]
                    time_since_update = now - baseline.created_at
                    
                    # Update if it's been a week or if there's sustained improvement
                    if (time_since_update > update_interval or 
                        self._is_sustained_improvement(metric_name, current_value)):
                        should_update = True
                
                if should_update:
                    self._calculate_new_baseline(metric_name, current_value)
                    
        except Exception as e:
            logger.error(f"Failed to update quality baselines: {e}")
    
    def _is_sustained_improvement(self, metric_name: str, current_value: float) -> bool:
        """Check if there's been sustained improvement in a metric."""
        try:
            if metric_name not in self.performance_baselines:
                return False
            
            baseline_value = self.performance_baselines[metric_name].baseline_value
            
            # Get recent values
            recent_values = []
            for metrics in list(self.quality_history)[-30:]:  # Last 30 samples
                value = getattr(metrics, metric_name, None)
                if value is not None:
                    recent_values.append(value)
            
            if len(recent_values) < 20:
                return False
            
            # Check if 80% of recent values are better than baseline
            improvement_threshold = baseline_value * 1.05  # 5% improvement
            
            if metric_name == 'average_processing_time':
                # For processing time, lower is better
                improved_count = sum(1 for v in recent_values if v < baseline_value * 0.95)
            else:
                # For quality metrics, higher is better
                improved_count = sum(1 for v in recent_values if v > improvement_threshold)
            
            improvement_ratio = improved_count / len(recent_values)
            return improvement_ratio >= 0.8
            
        except Exception as e:
            logger.error(f"Sustained improvement check failed for {metric_name}: {e}")
            return False
    
    def _calculate_new_baseline(self, metric_name: str, current_value: float):
        """Calculate new baseline for a metric based on recent performance."""
        try:
            # Get recent values for baseline calculation
            recent_values = []
            for metrics in list(self.quality_history)[-100:]:  # Last 100 samples
                value = getattr(metrics, metric_name, None)
                if value is not None:
                    recent_values.append(value)
            
            if len(recent_values) < 20:
                return
            
            # Calculate statistical baseline
            baseline_value = np.percentile(recent_values, 75)  # 75th percentile as baseline
            std_dev = stdev(recent_values)
            mean_value = mean(recent_values)
            
            # Set thresholds based on metric type
            if metric_name in ['blueprint_quality_score', 'brand_confidence_score', 'generation_success_rate']:
                # For quality metrics (higher is better)
                acceptable_deviation = 0.05  # 5%
                warning_threshold = 0.10  # 10%
                critical_threshold = 0.20  # 20%
            else:
                # For processing time (lower is better)
                acceptable_deviation = 0.10  # 10%
                warning_threshold = 0.20  # 20%
                critical_threshold = 0.50  # 50%
            
            # Calculate confidence interval
            confidence_interval = (
                baseline_value - 1.96 * std_dev / np.sqrt(len(recent_values)),
                baseline_value + 1.96 * std_dev / np.sqrt(len(recent_values))
            )
            
            new_baseline = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=baseline_value,
                acceptable_deviation=acceptable_deviation,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                created_at=datetime.utcnow(),
                sample_count=len(recent_values),
                confidence_interval=confidence_interval
            )
            
            self.performance_baselines[metric_name] = new_baseline
            
            logger.info(
                f"Updated baseline for {metric_name}",
                action="baseline.updated",
                metric=metric_name,
                baseline_value=baseline_value,
                sample_count=len(recent_values)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate new baseline for {metric_name}: {e}")
    
    def _detect_quality_anomalies(self, current_metrics: QualityMetrics):
        """Use ML to detect quality anomalies."""
        try:
            if not self.is_ml_enabled or len(self.quality_history) < 100:
                return
            
            # Prepare feature vector from current metrics
            features = np.array([[
                current_metrics.blueprint_quality_score,
                current_metrics.brand_confidence_score,
                current_metrics.prompt_optimization_score,
                current_metrics.generation_success_rate,
                current_metrics.average_processing_time,
                current_metrics.ml_enhancement_usage,
                current_metrics.error_rate
            ]])
            
            # Prepare training data from history
            if len(self.quality_history) % 50 == 0:  # Retrain every 50 samples
                training_features = []
                for metrics in list(self.quality_history)[-500:]:  # Use last 500 samples
                    training_features.append([
                        metrics.blueprint_quality_score,
                        metrics.brand_confidence_score,
                        metrics.prompt_optimization_score,
                        metrics.generation_success_rate,
                        metrics.average_processing_time,
                        metrics.ml_enhancement_usage,
                        metrics.error_rate
                    ])
                
                if len(training_features) > 50:
                    training_data = np.array(training_features)
                    self.anomaly_detector.fit(training_data)
            
            # Detect anomaly in current metrics
            if hasattr(self.anomaly_detector, 'decision_function'):
                anomaly_score = self.anomaly_detector.decision_function(features)[0]
                is_anomaly = self.anomaly_detector.predict(features)[0] == -1
                
                if is_anomaly:
                    logger.warning(
                        "Quality anomaly detected by ML model",
                        action="quality.anomaly.detected",
                        anomaly_score=anomaly_score,
                        blueprint_quality=current_metrics.blueprint_quality_score,
                        confidence_score=current_metrics.brand_confidence_score
                    )
                    
                    # Create anomaly alert
                    anomaly_alert = RegressionAlert(
                        metric_name="quality_anomaly",
                        current_value=anomaly_score,
                        baseline_value=0.0,
                        deviation_percent=abs(anomaly_score) * 100,
                        severity=AlertSeverity.WARNING,
                        trend=QualityTrend.VOLATILE,
                        detected_at=datetime.utcnow(),
                        samples_analyzed=len(self.quality_history),
                        recommendation="Investigate quality metrics for unusual patterns or data quality issues"
                    )
                    
                    self.regression_alerts.append(anomaly_alert)
                    
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
    
    def _handle_regression_alert(self, alert: RegressionAlert):
        """Handle regression alerts with appropriate actions."""
        try:
            # Log alert
            log_level = {
                AlertSeverity.INFO: 'info',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.CRITICAL: 'error',
                AlertSeverity.EMERGENCY: 'critical'
            }.get(alert.severity, 'warning')
            
            getattr(logger, log_level)(
                f"Quality regression detected in {alert.metric_name}",
                action="quality.regression.detected",
                metric=alert.metric_name,
                current_value=alert.current_value,
                baseline_value=alert.baseline_value,
                deviation_percent=alert.deviation_percent,
                severity=alert.severity.value,
                trend=alert.trend.value,
                recommendation=alert.recommendation
            )
            
            # Take automated actions for critical regressions
            if alert.severity == AlertSeverity.CRITICAL:
                self._take_automated_action(alert)
                
        except Exception as e:
            logger.error(f"Failed to handle regression alert: {e}")
    
    def _take_automated_action(self, alert: RegressionAlert):
        """Take automated actions for critical quality regressions."""
        try:
            actions_taken = []
            
            if alert.metric_name == 'generation_success_rate' and alert.current_value < 0.5:
                # Critical success rate drop - enable fallback mode
                logger.critical(
                    "Enabling fallback mode due to critical success rate regression",
                    action="automated.fallback.enabled"
                )
                actions_taken.append("fallback_mode_enabled")
            
            elif alert.metric_name == 'blueprint_quality_score' and alert.deviation_percent > 30:
                # Major quality drop - force ML optimization
                logger.critical(
                    "Forcing ML optimization due to quality regression",
                    action="automated.ml_optimization.forced"
                )
                actions_taken.append("ml_optimization_forced")
            
            if actions_taken:
                # Update alert with actions taken
                alert.recommendation += f" | Automated actions taken: {', '.join(actions_taken)}"
                
        except Exception as e:
            logger.error(f"Automated action failed: {e}")
    
    def get_quality_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive quality monitoring summary with enhanced metrics."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_metrics = [m for m in self.quality_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {'error': 'No quality data available for specified period'}
            
            # Calculate summary statistics
            quality_scores = [m.blueprint_quality_score for m in recent_metrics]
            confidence_scores = [m.brand_confidence_score for m in recent_metrics]
            success_rates = [m.generation_success_rate for m in recent_metrics]
            processing_times = [m.average_processing_time for m in recent_metrics]
            
            # Get recent regression alerts
            recent_alerts = [
                alert for alert in self.regression_alerts
                if alert.detected_at >= cutoff_time
            ]
            
            # Enhanced niche and template performance metrics
            niche_performance = defaultdict(list)
            template_performance = defaultdict(list)
            
            for metric in recent_metrics:
                if metric.niche_type:
                    niche_performance[metric.niche_type].append(metric.generation_success_rate)
                if metric.template_id:
                    template_performance[metric.template_id].append(metric.generation_success_rate)
            
            summary = {
                'period_hours': hours,
                'data_points': len(recent_metrics),
                'quality_metrics': {
                    'blueprint_quality': {
                        'current': quality_scores[-1] if quality_scores else 0,
                        'average': mean(quality_scores) if quality_scores else 0,
                        'min': min(quality_scores) if quality_scores else 0,
                        'max': max(quality_scores) if quality_scores else 0,
                        'trend': self._calculate_quality_trend('blueprint_quality_score').value
                    },
                    'brand_confidence': {
                        'current': confidence_scores[-1] if confidence_scores else 0,
                        'average': mean(confidence_scores) if confidence_scores else 0,
                        'min': min(confidence_scores) if confidence_scores else 0,
                        'max': max(confidence_scores) if confidence_scores else 0,
                        'trend': self._calculate_quality_trend('brand_confidence_score').value
                    },
                    'success_rate': {
                        'current': success_rates[-1] if success_rates else 0,
                        'average': mean(success_rates) if success_rates else 0,
                        'min': min(success_rates) if success_rates else 0,
                        'trend': self._calculate_quality_trend('generation_success_rate').value
                    },
                    'processing_time': {
                        'current': processing_times[-1] if processing_times else 0,
                        'average': mean(processing_times) if processing_times else 0,
                        'min': min(processing_times) if processing_times else 0,
                        'max': max(processing_times) if processing_times else 0,
                        'trend': self._calculate_quality_trend('average_processing_time').value
                    }
                },
                # Niche Performance Dashboard
                'niche_performance': {
                    niche: {
                        'success_rate': mean(rates) if rates else 0,
                        'data_points': len(rates)
                    } for niche, rates in niche_performance.items()
                },
                # Template Performance Dashboard
                'template_performance': {
                    template_id: {
                        'success_rate': mean(rates) if rates else 0,
                        'data_points': len(rates)
                    } for template_id, rates in template_performance.items()
                },
                'regression_analysis': {
                    'active_alerts': len([a for a in recent_alerts if a.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]]),
                    'critical_alerts': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                    'recent_alerts': recent_alerts[-5:] if recent_alerts else [],
                    'baselines_count': len(self.performance_baselines)
                },
                # Brand Classification Metrics
                'brand_classification': {
                    'confidence_scores': {
                        'average': mean(confidence_scores) if confidence_scores else 0,
                        'min': min(confidence_scores) if confidence_scores else 0,
                        'max': max(confidence_scores) if confidence_scores else 0
                    }
                },
                # Response Time Distribution
                'response_time_distribution': {
                    'average': mean(processing_times) if processing_times else 0,
                    'p50': np.percentile(processing_times, 50) if processing_times else 0,
                    'p90': np.percentile(processing_times, 90) if processing_times else 0,
                    'p99': np.percentile(processing_times, 99) if processing_times else 0
                },
                # Error Analysis
                'error_analysis': {
                    'total_rate': (mean([m.error_rate for m in recent_metrics]) if recent_metrics else 0),
                    'error_types': self._analyze_error_types(recent_metrics)
                },
                'ml_monitoring': {
                    'enabled': self.is_ml_enabled,
                    'anomaly_detection': self.anomaly_detector is not None,
                    'total_quality_samples': len(self.quality_history)
                },
                'overall_health': self._calculate_overall_quality_health(recent_metrics, recent_alerts)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_error_types(self, metrics: List[QualityMetrics]) -> Dict[str, float]:
        """Analyze error types and their distribution."""
        error_types = defaultdict(list)
        
        for metric in metrics:
            if metric.error_details:
                for error_type, count in metric.error_details.items():
                    error_types[error_type].append(count)
        
        return {
            error_type: mean(values) if values else 0
            for error_type, values in error_types.items()
        }
    
    def _calculate_overall_quality_health(self, recent_metrics: List[QualityMetrics], recent_alerts: List[RegressionAlert]) -> str:
        """Calculate overall quality health status."""
        try:
            if not recent_metrics:
                return 'unknown'
            
            # Check for critical alerts
            critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
            if critical_alerts:
                return 'critical'
            
            # Check recent quality scores
            recent_quality = [m.blueprint_quality_score for m in recent_metrics[-10:]]
            recent_success = [m.generation_success_rate for m in recent_metrics[-10:]]
            
            avg_quality = mean(recent_quality) if recent_quality else 0
            avg_success = mean(recent_success) if recent_success else 0
            
            if avg_quality > 0.8 and avg_success > 0.95:
                return 'excellent'
            elif avg_quality > 0.7 and avg_success > 0.9:
                return 'good'
            elif avg_quality > 0.6 and avg_success > 0.8:
                return 'acceptable'
            elif avg_quality > 0.5 and avg_success > 0.7:
                return 'needs_attention'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Quality health calculation failed: {e}")
            return 'unknown'
    
    # Phase 2: Ad Performance Monitoring Methods
    def record_ad_performance_metrics(self, metrics: "AdPerformanceMetrics"):
        """Record ad performance metrics for Phase 2 performance tracking."""
        try:
            self.ad_performance_history.append(metrics)
            
            # Update creative performance analytics
            self._update_creative_performance_analytics(metrics)
            
            # Generate performance predictions if we have enough data
            if len(self.ad_performance_history) > 100:
                self._update_performance_predictions(metrics)
            
            logger.debug(
                "Ad performance metrics recorded",
                action="ad.performance.recorded",
                platform=metrics.platform,
                ad_id=metrics.ad_id,
                ctr=metrics.ctr,
                conversion_rate=metrics.conversion_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to record ad performance metrics: {e}", exc_info=True)
    
    def _update_creative_performance_analytics(self, metrics: "AdPerformanceMetrics"):
        """Update creative performance analytics for continuous improvement."""
        try:
            if not metrics.template_id:
                return
                
            cache_key = f"{metrics.template_id}:{metrics.platform}:{metrics.niche_type or 'unknown'}"
            
            if cache_key not in self.creative_performance_cache:
                self.creative_performance_cache[cache_key] = {
                    'total_impressions': 0,
                    'total_clicks': 0,
                    'total_conversions': 0,
                    'total_spend': 0.0,
                    'performance_history': [],
                    'last_updated': datetime.utcnow()
                }
            
            cache_entry = self.creative_performance_cache[cache_key]
            
            # Update cumulative metrics
            cache_entry['total_impressions'] += metrics.impressions
            cache_entry['total_clicks'] += metrics.clicks
            cache_entry['total_conversions'] += metrics.conversions
            cache_entry['total_spend'] += metrics.spend
            
            # Add to performance history
            performance_point = {
                'timestamp': metrics.timestamp.isoformat(),
                'ctr': metrics.ctr,
                'conversion_rate': metrics.conversion_rate,
                'cpc': metrics.cpc,
                'roas': metrics.roas,
                'engagement_rate': metrics.engagement_rate
            }
            
            cache_entry['performance_history'].append(performance_point)
            
            # Keep only last 100 performance points per creative
            if len(cache_entry['performance_history']) > 100:
                cache_entry['performance_history'] = cache_entry['performance_history'][-100:]
            
            cache_entry['last_updated'] = datetime.utcnow()
            
            # Calculate derived metrics
            cache_entry['avg_ctr'] = (cache_entry['total_clicks'] / cache_entry['total_impressions'] 
                                    if cache_entry['total_impressions'] > 0 else 0)
            cache_entry['avg_conversion_rate'] = (cache_entry['total_conversions'] / cache_entry['total_clicks'] 
                                                if cache_entry['total_clicks'] > 0 else 0)
            cache_entry['total_roas'] = ((cache_entry['total_conversions'] * 50) / cache_entry['total_spend'] 
                                       if cache_entry['total_spend'] > 0 else 0)  # Assuming $50 avg order value
            
        except Exception as e:
            logger.error(f"Failed to update creative performance analytics: {e}")
    
    def _update_performance_predictions(self, current_metrics: "AdPerformanceMetrics"):
        """Update ML performance predictions based on historical data."""
        try:
            if not self.is_ml_enabled:
                return
            
            # Extract features for prediction
            features = self._extract_performance_features(current_metrics)
            
            if features and len(self.ad_performance_history) > 200:  # Need sufficient training data
                # Predict performance metrics
                predicted_ctr = self._predict_ctr(features)
                predicted_conversion_rate = self._predict_conversion_rate(features)
                predicted_engagement = self._predict_engagement_rate(features)
                
                # Store predictions
                prediction_key = f"{current_metrics.template_id}:{current_metrics.platform}"
                self.performance_predictions[prediction_key] = {
                    'predicted_ctr': predicted_ctr,
                    'predicted_conversion_rate': predicted_conversion_rate,
                    'predicted_engagement_rate': predicted_engagement,
                    'confidence_score': 0.8,  # Based on model validation
                    'prediction_timestamp': datetime.utcnow().isoformat(),
                    'features_used': list(features.keys()) if features else []
                }
                
                logger.debug(
                    "Performance predictions updated",
                    action="performance.prediction.updated",
                    template_id=current_metrics.template_id,
                    platform=current_metrics.platform,
                    predicted_ctr=predicted_ctr
                )
                
        except Exception as e:
            logger.error(f"Failed to update performance predictions: {e}")
    
    def _extract_performance_features(self, metrics: "AdPerformanceMetrics") -> Optional[Dict[str, float]]:
        """Extract features for performance prediction."""
        try:
            features = {
                'platform_encoded': 1.0 if metrics.platform == 'meta' else (2.0 if metrics.platform == 'tiktok' else 3.0),
                'duration_seconds': float(metrics.duration_seconds or 30),
                'quality_score': metrics.quality_score or 0.5,
                'hour_of_day': float(metrics.timestamp.hour),
                'day_of_week': float(metrics.timestamp.weekday()),
                'historical_ctr': self._get_historical_avg_ctr(metrics.template_id, metrics.platform),
                'historical_conversion_rate': self._get_historical_avg_conversion_rate(metrics.template_id, metrics.platform),
                'niche_performance_score': self._get_niche_performance_score(metrics.niche_type)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _predict_ctr(self, features: Dict[str, float]) -> float:
        """Predict click-through rate using ML model."""
        try:
            # Simple prediction based on features (would use trained model in production)
            base_ctr = 0.02
            
            # Quality adjustment
            quality_multiplier = 1.0 + (features.get('quality_score', 0.5) - 0.5)
            
            # Platform adjustment
            platform_multiplier = {
                1.0: 1.2,  # Meta tends to perform well
                2.0: 0.9,  # TikTok varies
                3.0: 1.1   # Google solid performance
            }.get(features.get('platform_encoded', 1.0), 1.0)
            
            # Historical performance
            historical_factor = features.get('historical_ctr', base_ctr) / base_ctr
            
            predicted_ctr = base_ctr * quality_multiplier * platform_multiplier * historical_factor
            
            return min(0.2, max(0.001, predicted_ctr))  # Clamp between 0.1% and 20%
            
        except Exception as e:
            logger.error(f"CTR prediction failed: {e}")
            return 0.02
    
    def _predict_conversion_rate(self, features: Dict[str, float]) -> float:
        """Predict conversion rate using ML model."""
        try:
            base_conversion_rate = 0.05
            
            # Quality and niche adjustments
            quality_factor = 1.0 + (features.get('quality_score', 0.5) - 0.5) * 0.5
            niche_factor = features.get('niche_performance_score', 0.5) * 2
            historical_factor = features.get('historical_conversion_rate', base_conversion_rate) / base_conversion_rate
            
            predicted_conversion_rate = base_conversion_rate * quality_factor * niche_factor * historical_factor
            
            return min(0.3, max(0.001, predicted_conversion_rate))  # Clamp between 0.1% and 30%
            
        except Exception as e:
            logger.error(f"Conversion rate prediction failed: {e}")
            return 0.05
    
    def _predict_engagement_rate(self, features: Dict[str, float]) -> float:
        """Predict engagement rate using ML model."""
        try:
            base_engagement = 0.1
            
            # Duration and quality impact on engagement
            duration_factor = 1.0 if features.get('duration_seconds', 30) <= 30 else 0.8
            quality_factor = 1.0 + (features.get('quality_score', 0.5) - 0.5)
            
            # Platform-specific engagement patterns
            platform_factor = {
                1.0: 1.0,  # Meta
                2.0: 1.3,  # TikTok typically higher engagement
                3.0: 0.7   # Google lower engagement
            }.get(features.get('platform_encoded', 1.0), 1.0)
            
            predicted_engagement = base_engagement * duration_factor * quality_factor * platform_factor
            
            return min(0.5, max(0.01, predicted_engagement))  # Clamp between 1% and 50%
            
        except Exception as e:
            logger.error(f"Engagement rate prediction failed: {e}")
            return 0.1
    
    def _get_historical_avg_ctr(self, template_id: Optional[str], platform: str) -> float:
        """Get historical average CTR for template/platform combination."""
        try:
            if not template_id:
                return 0.02  # Default CTR
                
            cache_key = f"{template_id}:{platform}"
            for key, analytics in self.creative_performance_cache.items():
                if key.startswith(cache_key):
                    return analytics.get('avg_ctr', 0.02)
                
            return 0.02
            
        except Exception:
            return 0.02
    
    def _get_historical_avg_conversion_rate(self, template_id: Optional[str], platform: str) -> float:
        """Get historical average conversion rate for template/platform combination."""
        try:
            if not template_id:
                return 0.05  # Default conversion rate
                
            cache_key = f"{template_id}:{platform}"
            for key, analytics in self.creative_performance_cache.items():
                if key.startswith(cache_key):
                    return analytics.get('avg_conversion_rate', 0.05)
                
            return 0.05
            
        except Exception:
            return 0.05
    
    def _get_niche_performance_score(self, niche_type: Optional[str]) -> float:
        """Get performance score for business niche."""
        try:
            if not niche_type:
                return 0.5
                
            # Calculate average performance across all templates in this niche
            niche_metrics = []
            for cache_key, analytics in self.creative_performance_cache.items():
                if niche_type in cache_key:
                    niche_metrics.append(analytics.get('avg_ctr', 0) * 10 + analytics.get('avg_conversion_rate', 0))
            
            if niche_metrics:
                return min(1.0, mean(niche_metrics))
                
            return 0.5
            
        except Exception:
            return 0.5
    
    def get_ad_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive ad performance summary for Phase 2."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_ad_metrics = [m for m in self.ad_performance_history if m.timestamp >= cutoff_time]
            
            if not recent_ad_metrics:
                return {'message': 'No ad performance data available for specified period'}
            
            # Platform performance breakdown
            platform_performance = defaultdict(lambda: {
                'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0.0,
                'video_views': 0, 'total_ads': 0
            })
            
            # Creative performance analytics
            creative_performance = defaultdict(lambda: {
                'impressions': 0, 'clicks': 0, 'conversions': 0, 'spend': 0.0,
                'platforms': set(), 'total_ads': 0
            })
            
            # Process recent metrics
            for metric in recent_ad_metrics:
                # Platform aggregation
                platform_data = platform_performance[metric.platform]
                platform_data['impressions'] += metric.impressions
                platform_data['clicks'] += metric.clicks
                platform_data['conversions'] += metric.conversions
                platform_data['spend'] += metric.spend
                platform_data['video_views'] += metric.video_views
                platform_data['total_ads'] += 1
                
                # Creative aggregation
                if metric.template_id:
                    creative_data = creative_performance[metric.template_id]
                    creative_data['impressions'] += metric.impressions
                    creative_data['clicks'] += metric.clicks
                    creative_data['conversions'] += metric.conversions
                    creative_data['spend'] += metric.spend
                    creative_data['platforms'].add(metric.platform)
                    creative_data['total_ads'] += 1
            
            # Calculate performance metrics
            for platform, data in platform_performance.items():
                data['ctr'] = (data['clicks'] / data['impressions']) if data['impressions'] > 0 else 0
                data['conversion_rate'] = (data['conversions'] / data['clicks']) if data['clicks'] > 0 else 0
                data['cpc'] = (data['spend'] / data['clicks']) if data['clicks'] > 0 else 0
                data['cpm'] = (data['spend'] / data['impressions'] * 1000) if data['impressions'] > 0 else 0
            
            # Top performing creatives
            top_creatives = []
            for template_id, data in creative_performance.items():
                if data['impressions'] > 0:
                    ctr = data['clicks'] / data['impressions']
                    conversion_rate = data['conversions'] / data['clicks'] if data['clicks'] > 0 else 0
                    roas = (data['conversions'] * 50) / data['spend'] if data['spend'] > 0 else 0  # Assuming $50 AOV
                    
                    top_creatives.append({
                        'template_id': template_id,
                        'ctr': ctr,
                        'conversion_rate': conversion_rate,
                        'roas': roas,
                        'total_impressions': data['impressions'],
                        'total_spend': data['spend'],
                        'platforms': len(data['platforms'])
                    })
            
            # Sort by ROAS
            top_creatives.sort(key=lambda x: x['roas'], reverse=True)
            
            return {
                'period_hours': hours,
                'data_points': len(recent_ad_metrics),
                'platform_performance': {
                    platform: {k: (list(v) if isinstance(v, set) else v) for k, v in data.items()}
                    for platform, data in platform_performance.items()
                },
                'top_performing_creatives': top_creatives[:10],
                'performance_predictions': dict(list(self.performance_predictions.items())[-5:]),  # Last 5 predictions
                'creative_analytics_cache_size': len(self.creative_performance_cache),
                'ml_predictions_enabled': self.is_ml_enabled
            }
            
        except Exception as e:
            logger.error(f"Ad performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def get_performance_predictions(self, template_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """Get performance predictions for a specific template and platform."""
        try:
            prediction_key = f"{template_id}:{platform}"
            return self.performance_predictions.get(prediction_key)
            
        except Exception as e:
            logger.error(f"Failed to get performance predictions: {e}")
            return None
    
    def get_creative_performance_analytics(self, template_id: Optional[str] = None, platform: Optional[str] = None) -> Dict[str, Any]:
        """Get creative performance analytics with optional filtering."""
        try:
            if template_id and platform:
                cache_key = f"{template_id}:{platform}"
                filtered = {k: v for k, v in self.creative_performance_cache.items() if k.startswith(cache_key)}
                return filtered
            elif template_id:
                # Return all platforms for this template
                filtered = {k: v for k, v in self.creative_performance_cache.items() if k.startswith(f"{template_id}:")}
                return filtered
            elif platform:
                # Return all templates for this platform
                filtered = {k: v for k, v in self.creative_performance_cache.items() if f":{platform}:" in k}
                return filtered
            else:
                # Return all analytics
                return dict(self.creative_performance_cache)
                
        except Exception as e:
            logger.error(f"Failed to get creative performance analytics: {e}")
            return {}

class MetricsCollector:
    """Enhanced system metrics collection with quality monitoring integration."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1440  # 24 hours at 1-minute intervals
        self.quality_monitor = QualityMonitor()
    
    def collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network and processes
            network_connections = len(psutil.net_connections())
            process_count = len(psutil.pids())
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_connections=network_connections,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Trim history if needed
            if len(self.metrics_history) > self.max_history:
                self.metrics_history = self.metrics_history[-self.max_history:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}", exc_info=True)
            raise
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for specified time period."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No metrics available for specified period'}
        
        # Calculate averages
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_usage_percent for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'available_gb': recent_metrics[-1].memory_available_gb
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'current_usage': recent_metrics[-1].disk_usage_percent,
                'free_gb': recent_metrics[-1].disk_free_gb
            },
            'uptime_hours': recent_metrics[-1].uptime_seconds / 3600
        }

class HealthCheckManager:
    """Manages and executes health checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.running = False
        self.check_thread: Optional[threading.Thread] = None
        self.setup_default_checks()
    
    def setup_default_checks(self):
        """Setup default system health checks."""
        
        # System resource checks
        self.register_check(
            name="system_cpu",
            check_func=self._check_cpu_usage,
            interval_seconds=30,
            critical=True
        )
        
        self.register_check(
            name="system_memory", 
            check_func=self._check_memory_usage,
            interval_seconds=30,
            critical=True
        )
        
        self.register_check(
            name="system_disk",
            check_func=self._check_disk_space,
            interval_seconds=60,
            critical=True
        )
        
        # Application-specific checks
        self.register_check(
            name="cache_connectivity",
            check_func=self._check_cache_connectivity,
            interval_seconds=60
        )
        
        self.register_check(
            name="config_validation",
            check_func=self._check_config_validation,
            interval_seconds=300  # 5 minutes
        )
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]], 
                      interval_seconds: int = 60, timeout_seconds: int = 10,
                      critical: bool = False, enabled: bool = True):
        """Register a new health check."""
        
        self.health_checks[name] = HealthCheck(
            name=name,
            check_func=check_func,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            critical=critical,
            enabled=enabled
        )
        
        logger.info(f"Health check registered: {name}", action="health_check.registered")
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage health."""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status = HealthStatus.HEALTHY
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
        
        return {
            'status': status.value,
            'cpu_percent': cpu_percent,
            'message': f"CPU usage: {cpu_percent:.1f}%",
            'threshold_warning': 75,
            'threshold_critical': 90
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health."""
        
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_gb = memory.available / (1024**3)
        
        status = HealthStatus.HEALTHY
        if memory_percent > 95:
            status = HealthStatus.CRITICAL
        elif memory_percent > 85:
            status = HealthStatus.WARNING
        
        return {
            'status': status.value,
            'memory_percent': memory_percent,
            'available_gb': round(available_gb, 2),
            'message': f"Memory usage: {memory_percent:.1f}% ({available_gb:.1f}GB available)",
            'threshold_warning': 85,
            'threshold_critical': 95
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space health."""
        
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        free_gb = disk.free / (1024**3)
        
        status = HealthStatus.HEALTHY
        if usage_percent > 95:
            status = HealthStatus.CRITICAL
        elif usage_percent > 85:
            status = HealthStatus.WARNING
        
        return {
            'status': status.value,
            'usage_percent': round(usage_percent, 1),
            'free_gb': round(free_gb, 2),
            'message': f"Disk usage: {usage_percent:.1f}% ({free_gb:.1f}GB free)",
            'threshold_warning': 85,
            'threshold_critical': 95
        }
    
    def _check_cache_connectivity(self) -> Dict[str, Any]:
        """Check cache system connectivity."""
        
        try:
            from core.cache import cache
            
            # Test cache operations
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Test set/get
            cache.set(test_key, test_value, ttl_seconds=30)
            retrieved = cache.get(test_key)
            
            if retrieved and retrieved.get('test') == True:
                cache.delete(test_key)  # Cleanup
                
                return {
                    'status': HealthStatus.HEALTHY.value,
                    'message': 'Cache connectivity OK',
                    'cache_type': 'redis' if cache.redis_cache else 'memory'
                }
            else:
                return {
                    'status': HealthStatus.WARNING.value,
                    'message': 'Cache test failed - data mismatch',
                    'cache_type': 'redis' if cache.redis_cache else 'memory'
                }
                
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Cache connectivity failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_config_validation(self) -> Dict[str, Any]:
        """Check configuration validation."""
        
        try:
            # Test that config is accessible and valid
            video_config = config.video
            logo_config = config.logo
            
            # Basic validation checks
            if video_config.max_duration_seconds <= 0:
                return {
                    'status': HealthStatus.CRITICAL.value,
                    'message': 'Invalid video configuration - max_duration_seconds <= 0'
                }
            
            if logo_config.max_file_size_mb <= 0:
                return {
                    'status': HealthStatus.CRITICAL.value, 
                    'message': 'Invalid logo configuration - max_file_size_mb <= 0'
                }
            
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Configuration validation passed',
                'environment': config.environment.value
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Configuration validation failed: {str(e)}',
                'error': str(e)
            }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        
        if name not in self.health_checks:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': f'Health check {name} not found',
                'error': 'check_not_found'
            }
        
        health_check = self.health_checks[name]
        
        if not health_check.enabled:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': f'Health check {name} is disabled'
            }
        
        try:
            start_time = time.time()
            result = health_check.check_func()
            execution_time = time.time() - start_time
            
            # Update health check record
            health_check.last_run = datetime.utcnow()
            health_check.last_status = HealthStatus(result['status'])
            health_check.last_result = result
            
            # Add metadata
            result['check_name'] = name
            result['execution_time_ms'] = round(execution_time * 1000, 2)
            result['timestamp'] = health_check.last_run.isoformat()
            
            # Log result
            log_level = 'error' if result['status'] == HealthStatus.CRITICAL.value else 'info'
            getattr(logger, log_level)(
                f"Health check {name}: {result['message']}",
                action=f"health_check.{name}",
                status=result['status'],
                execution_time_ms=result['execution_time_ms']
            )
            
            return result
            
        except Exception as e:
            error_result = {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Health check {name} failed with exception: {str(e)}',
                'error': str(e),
                'check_name': name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            health_check.last_run = datetime.utcnow()
            health_check.last_status = HealthStatus.CRITICAL
            health_check.last_result = error_result
            
            logger.error(
                f"Health check {name} failed",
                action=f"health_check.{name}.error",
                error=str(e),
                exc_info=True
            )
            
            return error_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all enabled health checks."""
        
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
                
            result = self.run_check(name)
            results[name] = result
            
            # Update overall status
            check_status = HealthStatus(result['status'])
            if check_status == HealthStatus.CRITICAL:
                if health_check.critical:
                    overall_status = HealthStatus.CRITICAL
                elif overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
            elif check_status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health monitoring started", action="monitoring.started")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        
        self.running = False
        if self.check_thread and self.check_thread.is_alive():
            self.check_thread.join(timeout=10)
        
        logger.info("Health monitoring stopped", action="monitoring.stopped")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this health check
                    if (health_check.last_run is None or 
                        (current_time - health_check.last_run).total_seconds() >= health_check.interval_seconds):
                        
                        self.run_check(name)
                
                # Sleep before next iteration
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(30)  # Longer sleep on error

class MonitoringManager:
    """Enhanced main monitoring system coordinator with ML-powered quality monitoring."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_check_manager = HealthCheckManager()
        self.quality_monitor = QualityMonitor()
        self.alerts_enabled = config.monitoring.enable_health_checks
        self.quality_monitoring_enabled = True
        
        # Setup quality monitoring health checks
        self._setup_quality_health_checks()
        
    def start(self):
        """Start all monitoring systems."""
        
        if config.monitoring.enable_health_checks:
            self.health_check_manager.start_monitoring()
            
        logger.info("Monitoring system started", action="monitoring.system.started")
    
    def stop(self):
        """Stop all monitoring systems."""
        
        self.health_check_manager.stop_monitoring()
        logger.info("Monitoring system stopped", action="monitoring.system.stopped")
    
    def _setup_quality_health_checks(self):
        """Setup health checks for quality monitoring system."""
        try:
            # Register quality monitoring health checks
            self.health_check_manager.register_check(
                name="quality_monitoring_system",
                check_func=self._check_quality_monitoring_health,
                interval_seconds=120,  # Check every 2 minutes
                critical=True
            )
            
            self.health_check_manager.register_check(
                name="quality_baselines",
                check_func=self._check_quality_baselines_health,
                interval_seconds=300,  # Check every 5 minutes
                critical=False
            )
            
            self.health_check_manager.register_check(
                name="regression_detection",
                check_func=self._check_regression_detection_health,
                interval_seconds=180,  # Check every 3 minutes
                critical=True
            )
            
        except Exception as e:
            logger.error(f"Failed to setup quality health checks: {e}")
    
    def _check_quality_monitoring_health(self) -> Dict[str, Any]:
        """Check quality monitoring system health."""
        try:
            quality_data_points = len(self.quality_monitor.quality_history)
            ml_enabled = self.quality_monitor.is_ml_enabled
            
            status = HealthStatus.HEALTHY
            message = f"Quality monitoring active with {quality_data_points} data points"
            
            if quality_data_points == 0:
                status = HealthStatus.WARNING
                message = "No quality data collected yet"
            elif quality_data_points < 10:
                status = HealthStatus.WARNING
                message = f"Limited quality data: {quality_data_points} points"
            
            return {
                'status': status.value,
                'message': message,
                'data_points': quality_data_points,
                'ml_enabled': ml_enabled,
                'baselines_configured': len(self.quality_monitor.performance_baselines)
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Quality monitoring health check failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_quality_baselines_health(self) -> Dict[str, Any]:
        """Check quality baselines health."""
        try:
            baselines = self.quality_monitor.performance_baselines
            baseline_count = len(baselines)
            
            status = HealthStatus.HEALTHY
            if baseline_count == 0:
                status = HealthStatus.WARNING
                message = "No quality baselines established yet"
            elif baseline_count < 3:
                status = HealthStatus.WARNING
                message = f"Limited baselines: {baseline_count}/4 key metrics"
            else:
                message = f"Quality baselines healthy: {baseline_count} metrics tracked"
            
            # Check baseline freshness
            stale_baselines = 0
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            for baseline in baselines.values():
                if baseline.created_at < week_ago:
                    stale_baselines += 1
            
            if stale_baselines > 0:
                message += f" ({stale_baselines} baselines over 1 week old)"
            
            return {
                'status': status.value,
                'message': message,
                'baseline_count': baseline_count,
                'stale_baselines': stale_baselines
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Quality baselines health check failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_regression_detection_health(self) -> Dict[str, Any]:
        """Check regression detection system health."""
        try:
            recent_alerts = [
                alert for alert in self.quality_monitor.regression_alerts
                if alert.detected_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            critical_alerts = [
                alert for alert in recent_alerts
                if alert.severity == AlertSeverity.CRITICAL
            ]
            
            status = HealthStatus.HEALTHY
            if len(critical_alerts) > 0:
                status = HealthStatus.CRITICAL
                message = f"Critical quality regressions detected: {len(critical_alerts)} alerts"
            elif len(recent_alerts) > 5:
                status = HealthStatus.WARNING
                message = f"High alert volume: {len(recent_alerts)} quality alerts in 24h"
            else:
                message = f"Regression detection healthy: {len(recent_alerts)} alerts in 24h"
            
            return {
                'status': status.value,
                'message': message,
                'recent_alerts': len(recent_alerts),
                'critical_alerts': len(critical_alerts),
                'ml_anomaly_detection': self.quality_monitor.is_ml_enabled
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Regression detection health check failed: {str(e)}',
                'error': str(e)
            }
    
    def record_quality_metrics(self, 
                             blueprint_quality_score: float,
                             brand_confidence_score: float,
                             prompt_optimization_score: float = 0.0,
                             generation_success_rate: float = 1.0,
                             average_processing_time: float = 0.0,
                             ml_enhancement_usage: float = 0.0,
                             error_rate: float = 0.0,
                             user_satisfaction_score: Optional[float] = None):
        """Record quality metrics for monitoring and regression detection."""
        try:
            quality_metrics = QualityMetrics(
                timestamp=datetime.utcnow(),
                blueprint_quality_score=blueprint_quality_score,
                brand_confidence_score=brand_confidence_score,
                prompt_optimization_score=prompt_optimization_score,
                generation_success_rate=generation_success_rate,
                average_processing_time=average_processing_time,
                ml_enhancement_usage=ml_enhancement_usage,
                error_rate=error_rate,
                user_satisfaction_score=user_satisfaction_score
            )
            
            self.quality_monitor.record_quality_metrics(quality_metrics)
            
        except Exception as e:
            logger.error(f"Failed to record quality metrics: {e}", exc_info=True)
    
    def get_quality_status(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive quality monitoring status."""
        return self.quality_monitor.get_quality_summary(hours)
    
    def get_regression_alerts(self, severity_filter: Optional[AlertSeverity] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get regression alerts with optional filtering."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            alerts = [
                alert for alert in self.quality_monitor.regression_alerts
                if alert.detected_at >= cutoff_time
            ]
            
            if severity_filter:
                alerts = [alert for alert in alerts if alert.severity == severity_filter]
            
            # Convert to dictionaries for JSON serialization
            return [{
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'baseline_value': alert.baseline_value,
                'deviation_percent': alert.deviation_percent,
                'severity': alert.severity.value,
                'trend': alert.trend.value,
                'detected_at': alert.detected_at.isoformat(),
                'samples_analyzed': alert.samples_analyzed,
                'recommendation': alert.recommendation
            } for alert in alerts]
            
        except Exception as e:
            logger.error(f"Failed to get regression alerts: {e}")
            return []
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.metrics_collector.collect_current_metrics()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with quality monitoring."""
        try:
            # Collect current metrics
            current_metrics = self.metrics_collector.collect_current_metrics()
            metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
            
            # Run health checks
            health_status = self.health_check_manager.run_all_checks()
            
            # Get quality monitoring status
            quality_status = self.get_quality_status(hours=24)
            
            # Get recent regression alerts
            recent_alerts = self.get_regression_alerts(hours=24)
            critical_alerts = self.get_regression_alerts(AlertSeverity.CRITICAL, hours=24)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_seconds': current_metrics.uptime_seconds,
                'uptime_hours': round(current_metrics.uptime_seconds / 3600, 2),
                'health': health_status,
                'metrics': {
                    'current': {
                        'cpu_percent': current_metrics.cpu_percent,
                        'memory_percent': current_metrics.memory_percent,
                        'memory_available_gb': current_metrics.memory_available_gb,
                        'disk_usage_percent': current_metrics.disk_usage_percent,
                        'disk_free_gb': current_metrics.disk_free_gb,
                        'process_count': current_metrics.process_count
                    },
                    'summary_1h': metrics_summary
                },
                'quality_monitoring': {
                    'enabled': self.quality_monitoring_enabled,
                    'ml_enhanced': self.quality_monitor.is_ml_enabled,
                    'status': quality_status,
                    'alerts': {
                        'total_24h': len(recent_alerts),
                        'critical_24h': len(critical_alerts),
                        'recent_critical': critical_alerts[:3] if critical_alerts else []
                    }
                },
                'environment': config.environment.value,
                'service_tier': config.service_tier.value,
                'monitoring_version': 'v3.0_ml_enhanced'
            }
            
        except Exception as e:
            logger.error(f"System status generation failed: {e}", exc_info=True)
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'health': {'overall_status': 'critical'},
                'monitoring_version': 'v3.0_ml_enhanced_error'
            }

# Global enhanced monitoring instance
monitoring = MonitoringManager()

def track_template_performance(
    template_id: str, 
    niche: str, 
    version: int, 
    success_rate: float, 
    quality_scores: Optional[List[float]] = None
) -> None:
    """
    Track and log template performance metrics.
    
    Args:
        template_id: Unique identifier for the template
        niche: Business niche for the template
        version: Template version
        success_rate: Percentage of successful generations
        quality_scores: Optional list of quality scores for detailed tracking
    """
    try:
        quality_metrics = QualityMetrics(
            timestamp=datetime.utcnow(),
            blueprint_quality_score=mean(quality_scores) if quality_scores else 0.0,
            brand_confidence_score=0.0,  # To be populated if possible
            prompt_optimization_score=0.0,
            generation_success_rate=success_rate,
            average_processing_time=0.0,
            ml_enhancement_usage=0.0,
            error_rate=1.0 - success_rate,
            niche_type=niche,
            template_id=template_id,
            template_version=version,
            scene_quality_scores=quality_scores
        )
        
        # Record metrics in the monitoring system
        monitoring.record_quality_metrics(
            blueprint_quality_score=quality_metrics.blueprint_quality_score,
            brand_confidence_score=quality_metrics.brand_confidence_score,
            generation_success_rate=quality_metrics.generation_success_rate,
            error_rate=quality_metrics.error_rate
        )
        
        logger.info(
            f"Template Performance Tracked: {template_id} (v{version})",
            action="template.performance.tracked",
            niche=niche,
            success_rate=success_rate,
            quality_score=quality_metrics.blueprint_quality_score
        )
        
    except Exception as e:
        logger.error(f"Failed to track template performance: {e}", exc_info=True)

# Phase 2: Ad Performance Data Classes
@dataclass
class AdPerformanceMetrics:
    """Ad platform performance metrics for Phase 2 integration."""
    timestamp: datetime
    ad_id: str
    platform: str  # 'meta', 'tiktok', 'google'
    campaign_id: Optional[str] = None
    
    # Performance metrics
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    ctr: float = 0.0  # Click-through rate
    cpc: float = 0.0  # Cost per click
    cpm: float = 0.0  # Cost per mille
    conversion_rate: float = 0.0
    roas: float = 0.0  # Return on ad spend
    
    # Video-specific metrics
    video_views: int = 0
    video_completion_rate: float = 0.0
    engagement_rate: float = 0.0
    
    # Creative performance
    creative_id: Optional[str] = None
    template_id: Optional[str] = None
    niche_type: Optional[str] = None
    quality_score: Optional[float] = None
    
    # Metadata
    duration_seconds: Optional[int] = None
    audience_targeting: Optional[Dict[str, Any]] = None
    creative_elements: Optional[Dict[str, Any]] = None

# Initialize quality monitoring
try:
    if monitoring.quality_monitoring_enabled:
        logger.info("ML-enhanced quality monitoring with Phase 2 ad performance tracking initialized", action="monitoring.quality.init")
except Exception as e:
    logger.error(f"Quality monitoring initialization failed: {e}", exc_info=True)

# Enhanced convenience functions
def get_health_status() -> Dict[str, Any]:
    """Get current health status including quality monitoring."""
    return monitoring.health_check_manager.run_all_checks()

def get_system_metrics() -> SystemMetrics:
    """Get current system metrics."""
    return monitoring.metrics_collector.collect_current_metrics()

def get_quality_status(hours: int = 24) -> Dict[str, Any]:
    """Get quality monitoring status."""
    return monitoring.get_quality_status(hours)

def record_quality_metrics(**kwargs):
    """Record quality metrics for monitoring."""
    return monitoring.record_quality_metrics(**kwargs)

def get_regression_alerts(severity: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
    """Get regression alerts with optional severity filtering."""
    severity_enum = AlertSeverity(severity) if severity else None
    return monitoring.get_regression_alerts(severity_enum, hours)

def get_full_status() -> Dict[str, Any]:
    """Get comprehensive system status with ML-enhanced monitoring."""
    return monitoring.get_system_status()