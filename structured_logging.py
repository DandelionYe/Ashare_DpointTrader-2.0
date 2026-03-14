"""
结构化日志配置。

统一日志格式，输出到文件和控制台，支持日志级别和结构化数据。
"""

from __future__ import annotations

import logging
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# =========================================================
# Custom JSON Formatter for Structured Logging
# =========================================================
class StructuredFormatter(logging.Formatter):
    """结构化日志格式器，输出 JSON 格式"""

    def format(self, record: logging.LogRecord) -> str:
        """格式化为 JSON 字符串"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ["msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "message", "name", "exc_info",
                          "exc_text", "stack_info", "taskName"]:
                try:
                    json.dumps(value)  # 检查是否可序列化
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """控制台日志格式器，人类可读格式"""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        color = self.COLORS.get(level, "")
        
        message = f"{timestamp} | {color}{level:<8}{self.RESET} | {record.name} | {record.getMessage()}"
        
        # 添加额外字段
        extras = []
        for key, value in record.__dict__.items():
            if key not in ["msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "message", "name"]:
                extras.append(f"{key}={value}")
        
        if extras:
            message += f" | {', '.join(extras)}"
        
        return message


# =========================================================
# Logger Configuration
# =========================================================
def setup_logger(
    name: str = "dpoint_trader",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    配置结构化日志器。

    参数：
        name           : 日志器名称
        level          : 日志级别
        log_dir        : 日志文件目录（None 表示不输出到文件）
        console_output : 是否输出到控制台
        file_output    : 是否输出到文件
        json_format    : 是否使用 JSON 格式（文件日志默认 JSON）

    返回：
        配置好的 Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的 handler
    logger.handlers.clear()

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if json_format:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())
        
        logger.addHandler(console_handler)

    # 文件处理器
    if file_output and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        
        # 文件日志默认使用 JSON 格式
        file_handler.setFormatter(StructuredFormatter())
        
        logger.addHandler(file_handler)

    return logger


# =========================================================
# Convenience Functions
# =========================================================
def get_logger(name: str = "dpoint_trader") -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


def log_extra(logger: logging.Logger, level: int, message: str, **kwargs):
    """记录带额外字段的日志"""
    logger.log(level, message, extra=kwargs)


def debug_extra(logger: logging.Logger, message: str, **kwargs):
    """记录 DEBUG 级别的结构化日志"""
    log_extra(logger, logging.DEBUG, message, **kwargs)


def info_extra(logger: logging.Logger, message: str, **kwargs):
    """记录 INFO 级别的结构化日志"""
    log_extra(logger, logging.INFO, message, **kwargs)


def warning_extra(logger: logging.Logger, message: str, **kwargs):
    """记录 WARNING 级别的结构化日志"""
    log_extra(logger, logging.WARNING, message, **kwargs)


def error_extra(logger: logging.Logger, message: str, **kwargs):
    """记录 ERROR 级别的结构化日志"""
    log_extra(logger, logging.ERROR, message, **kwargs)


# =========================================================
# Context Manager for Logging
# =========================================================
class log_context:
    """日志上下文管理器，自动记录开始/结束和时间"""

    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.extra_info = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        info_extra(
            self.logger,
            f"Starting: {self.operation}",
            operation=self.operation,
            status="started",
            **self.extra_info
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            info_extra(
                self.logger,
                f"Completed: {self.operation}",
                operation=self.operation,
                status="completed",
                elapsed_seconds=elapsed,
                **self.extra_info
            )
        else:
            error_extra(
                self.logger,
                f"Failed: {self.operation}",
                operation=self.operation,
                status="failed",
                elapsed_seconds=elapsed,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.extra_info
            )
        
        return False  # 不抑制异常


# =========================================================
# Performance Logging
# =========================================================
class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timings: Dict[str, datetime] = {}

    def start(self, operation: str, **kwargs):
        """记录操作开始"""
        self.timings[operation] = datetime.now()
        debug_extra(self.logger, f"Timing start: {operation}", **kwargs)

    def end(self, operation: str, **kwargs):
        """记录操作结束和耗时"""
        if operation not in self.timings:
            self.logger.warning(f"No start time recorded for: {operation}")
            return
        
        elapsed = (datetime.now() - self.timings[operation]).total_seconds()
        del self.timings[operation]
        
        info_extra(
            self.logger,
            f"Timing end: {operation}",
            operation=operation,
            elapsed_seconds=elapsed,
            **kwargs
        )
        return elapsed


# =========================================================
# Default Logger Instance
# =========================================================
default_logger = setup_logger(
    name="dpoint_trader",
    level=logging.INFO,
    log_dir="./logs",
    console_output=True,
    file_output=True,
    json_format=False,  # 控制台使用人类可读格式
)
