# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

from .fa_config import build_report_path

_ERROR_KEYWORDS = (
    "错误",
    "出错",
    "失败",
    "[ERROR]",
)

_WARN_KEYWORDS = (
    "警告",
    "[WARN]",
)

_INFO_KEYWORDS = (
    "[INFO]",
    "[STEP]",
    "[OK]",
    "[DEBUG]",
)


class Logger:
    """简单日志记录器：默认仅将错误写入日志，可通过环境变量启用警告。"""

    def __init__(self, log_file=None, log_all=None, include_warnings=None):
        """初始化日志记录器

        Args:
            log_file: 日志文件路径，如果为 None 则自动按时间戳生成
            log_all: 是否记录全部输出，默认读取环境变量 FA_LOG_VERBOSE
            include_warnings: 是否记录警告日志，默认读取环境变量 FA_LOG_WARNINGS
        """
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'因子分析日志_{timestamp}.txt'

        log_file = build_report_path(log_file)

        self.log_file = log_file
        self.terminal = sys.stdout  # 保存原始终端输出
        env_verbose = os.environ.get("FA_LOG_VERBOSE")
        env_warning = os.environ.get("FA_LOG_WARNINGS")
        self.log_all = log_all if log_all is not None else env_verbose == "1"
        self.include_warnings = (
            include_warnings
            if include_warnings is not None
            else env_warning == "1"
        )
        self._buffer = ""

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"因子分析日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

    def _line_contains_keyword(self, line: str) -> bool:
        """检查单行是否需要写入到日志。"""
        if self.log_all:
            return True

        content = line.strip()
        if not content:
            return False

        if any(keyword in content for keyword in _ERROR_KEYWORDS):
            return True

        # “异常”一词既可能出现在警告中，也可能表示真正的错误。
        # 仅当不包含“警告”前缀时，才将其视作错误输出。
        if "异常" in content and "警告" not in content:
            return True

        if any(keyword in content for keyword in _INFO_KEYWORDS):
            return True

        if self.include_warnings and any(keyword in content for keyword in _WARN_KEYWORDS):
            return True

        return False

    def _drain_buffer(self, force: bool = False):
        """将缓冲区中带换行的内容输出到日志文件。"""
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._log_line(line.rstrip('\r'))

        if force and self._buffer:
            self._log_line(self._buffer.rstrip('\r'))
            self._buffer = ""

    def _log_line(self, line: str):
        if not self._line_contains_keyword(line):
            return

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{line}\n")

    def write(self, message):
        """同时输出到终端，并在需要时写入日志文件"""
        self.terminal.write(message)
        self._buffer += message
        self._drain_buffer()

    def flush(self):
        """刷新输出"""
        self.terminal.flush()
        self._drain_buffer()

    def close(self):
        """关闭日志记录器并恢复标准输出"""
        self._drain_buffer(force=True)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n日志记录结束 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

        sys.stdout = self.terminal
