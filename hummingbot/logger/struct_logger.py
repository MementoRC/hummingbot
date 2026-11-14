# Compat re-export: superseded by hb-logger (logger.struct_logger).
from logger.struct_logger import EVENT_LOG_LEVEL, METRICS_LOG_LEVEL, StructLogger, StructLogRecord

__all__ = ["EVENT_LOG_LEVEL", "METRICS_LOG_LEVEL", "StructLogger", "StructLogRecord"]
