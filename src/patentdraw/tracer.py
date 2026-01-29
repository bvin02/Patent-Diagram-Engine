"""
Hierarchical runtime tracing for the Patent Draw pipeline.

Provides structured, nested logging with timing information to help debug
pipeline execution without stepping through code.
"""

import functools
import hashlib
import io
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime


class TracerConfig:
    """Configuration for the tracer."""
    
    def __init__(self):
        self.enabled = False
        self.level = "INFO"
        self.file_path = None
        self.json_output = False
        self._file_handle = None
    
    def configure(self, enabled=False, level="INFO", file_path=None, json_output=False):
        """Configure tracer settings."""
        self.enabled = enabled
        self.level = level.upper()
        self.file_path = file_path
        self.json_output = json_output
        
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        
        if file_path and enabled:
            self._file_handle = open(file_path, "w", encoding="utf-8")
    
    def close(self):
        """Close file handle if open."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


class Tracer:
    """
    Hierarchical tracer for structured pipeline logging.
    
    Supports nested spans with timing, argument summarization, and
    multiple output formats (text, JSON).
    """
    
    LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}
    
    def __init__(self):
        self.config = TracerConfig()
        self._depth = 0
        self._span_stack = []
    
    def _should_log(self, level):
        """Check if this level should be logged."""
        if not self.config.enabled:
            return False
        return self.LEVELS.get(level, 2) <= self.LEVELS.get(self.config.level, 2)
    
    def _format_timestamp(self):
        """Format current time as HH:MM:SS.mmm."""
        now = datetime.now()
        return now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    
    def _indent(self):
        """Get indentation string based on depth."""
        return "  " * self._depth
    
    def _write(self, level, module, func, message, meta=None):
        """Write a log line."""
        if not self._should_log(level):
            return
        
        timestamp = self._format_timestamp()
        indent = self._indent()
        location = f"{module}:{func}" if func else module
        
        text_line = f"{timestamp} {level:<5} {indent}{location}  {message}"
        
        print(text_line, file=sys.stderr)
        
        if self.config._file_handle:
            self.config._file_handle.write(text_line + "\n")
            self.config._file_handle.flush()
        
        if self.config.json_output:
            json_record = {
                "timestamp": timestamp,
                "level": level,
                "depth": self._depth,
                "module": module,
                "function": func,
                "message": message,
                "meta": meta or {},
            }
            json_line = json.dumps(json_record)
            print(json_line, file=sys.stderr)
            if self.config._file_handle:
                self.config._file_handle.write(json_line + "\n")
    
    @contextmanager
    def span(self, name, module="", **meta):
        """
        Context manager for a traced span.
        
        Logs start and end with timing information.
        """
        if not self.config.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        meta_str = " ".join(f"{k}={summarize(v)}" for k, v in meta.items())
        self._write("INFO", module, name, f"start {meta_str}".strip())
        self._depth += 1
        self._span_stack.append((name, module, start_time))
        
        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            elapsed = (time.perf_counter() - start_time) * 1000
            self._depth -= 1
            self._write("ERROR", module, name, f"failed dt={elapsed:.0f}ms error={type(e).__name__}: {str(e)[:100]}")
            raise
        finally:
            if not error_occurred:
                elapsed = (time.perf_counter() - start_time) * 1000
                self._depth -= 1
                self._span_stack.pop()
                self._write("INFO", module, name, f"end ok dt={elapsed:.0f}ms")
    
    def event(self, message, level="INFO", **meta):
        """Log a one-off event within the current span."""
        if not self._should_log(level):
            return
        
        module = ""
        func = ""
        if self._span_stack:
            func, module, _ = self._span_stack[-1]
        
        meta_str = " ".join(f"{k}={summarize(v)}" for k, v in meta.items())
        full_message = f"{message} {meta_str}".strip()
        self._write(level, module, func, full_message, meta)


def summarize(obj, max_len=200):
    """
    Summarize an object for logging.
    
    Returns a compact string representation that never exceeds max_len chars.
    Handles numpy arrays, images, lists, dicts, strings, pydantic models,
    shapely geometries, and networkx graphs.
    """
    try:
        result = _summarize_impl(obj)
        if len(result) > max_len:
            return result[:max_len - 3] + "..."
        return result
    except Exception:
        return f"<{type(obj).__name__}>"


def _summarize_impl(obj):
    """Implementation of summarize without length capping."""
    if obj is None:
        return "None"
    
    type_name = type(obj).__name__
    
    # numpy arrays
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            shape_str = "x".join(str(s) for s in obj.shape)
            dtype_str = str(obj.dtype)
            if obj.size > 0 and obj.size < 1000:
                h = hashlib.md5(obj.tobytes()).hexdigest()[:8]
            else:
                h = hashlib.md5(str(obj.shape).encode()).hexdigest()[:8]
            return f"ndarray({dtype_str},{shape_str},h={h})"
    except ImportError:
        pass
    
    # shapely geometries
    try:
        from shapely.geometry.base import BaseGeometry
        if isinstance(obj, BaseGeometry):
            bounds = obj.bounds
            bounds_str = ",".join(f"{b:.1f}" for b in bounds)
            return f"{type_name}(bounds=[{bounds_str}])"
    except ImportError:
        pass
    
    # networkx graphs
    try:
        import networkx as nx
        if isinstance(obj, (nx.Graph, nx.DiGraph)):
            return f"{type_name}(nodes={obj.number_of_nodes()},edges={obj.number_of_edges()})"
    except ImportError:
        pass
    
    # pydantic models
    try:
        from pydantic import BaseModel
        if isinstance(obj, BaseModel):
            fields = list(type(obj).model_fields.keys())[:3]
            return f"{type_name}(fields={fields}...)"
    except ImportError:
        pass
    
    # strings
    if isinstance(obj, str):
        if len(obj) > 50:
            h = hashlib.md5(obj.encode()).hexdigest()[:8]
            return f"str(len={len(obj)},h={h})"
        return repr(obj)
    
    # bytes
    if isinstance(obj, bytes):
        h = hashlib.md5(obj).hexdigest()[:8]
        return f"bytes(len={len(obj)},h={h})"
    
    # lists and tuples
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return f"{type_name}(len=0)"
        first_type = type(obj[0]).__name__ if obj else "?"
        return f"{type_name}(len={len(obj)},first={first_type})"
    
    # dicts
    if isinstance(obj, dict):
        keys = list(obj.keys())[:5]
        keys_str = ",".join(str(k) for k in keys)
        return f"dict(len={len(obj)},keys=[{keys_str}])"
    
    # numbers
    if isinstance(obj, (int, float)):
        return str(obj)
    
    # fallback
    return f"<{type_name}>"


def trace(label=None, arg_names=None):
    """
    Decorator to trace function execution.
    
    Wraps a function in a span that logs start/end with timing.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _tracer.config.enabled:
                return func(*args, **kwargs)
            
            func_module = func.__module__.split(".")[-1] if func.__module__ else ""
            func_name = label or func.__name__
            
            meta = {}
            if arg_names:
                for name in arg_names:
                    if name in kwargs:
                        meta[name] = kwargs[name]
            
            with _tracer.span(func_name, module=func_module, **meta):
                result = func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator


# Global tracer instance
_tracer = Tracer()


def get_tracer():
    """Get the global tracer instance."""
    return _tracer


def configure_tracer(enabled=False, level="INFO", file_path=None, json_output=False):
    """Configure the global tracer."""
    _tracer.config.configure(
        enabled=enabled,
        level=level,
        file_path=file_path,
        json_output=json_output,
    )
