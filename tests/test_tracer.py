"""Tests for the tracer module."""

import numpy as np
import pytest


class TestSummarize:
    """Tests for object summarization."""
    
    def test_numpy_array_summary(self):
        """Test that numpy arrays are summarized with shape and type."""
        from patentdraw.tracer import summarize
        
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        summary = summarize(arr)
        
        assert "ndarray" in summary
        assert "100x200x3" in summary or "100, 200, 3" in summary
        assert "uint8" in summary
    
    def test_summary_capped_length(self):
        """Test that summary never exceeds max length."""
        from patentdraw.tracer import summarize
        
        # Large dict
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        summary = summarize(large_dict, max_len=200)
        
        assert len(summary) <= 200
    
    def test_list_summary(self):
        """Test list summarization."""
        from patentdraw.tracer import summarize
        
        lst = [1, 2, 3, 4, 5]
        summary = summarize(lst)
        
        assert "list" in summary
        assert "len=5" in summary
    
    def test_string_summary(self):
        """Test long string summarization."""
        from patentdraw.tracer import summarize
        
        long_string = "a" * 1000
        summary = summarize(long_string)
        
        assert "str" in summary
        assert "len=1000" in summary
        assert len(summary) <= 200
    
    def test_none_summary(self):
        """Test None summarization."""
        from patentdraw.tracer import summarize
        
        assert summarize(None) == "None"
    
    def test_pydantic_model_summary(self):
        """Test Pydantic model summarization."""
        from patentdraw.tracer import summarize
        from patentdraw.models import Stroke
        
        stroke = Stroke(
            stroke_id="test",
            polyline=[[0, 0], [10, 10]],
        )
        summary = summarize(stroke)
        
        assert "Stroke" in summary


class TestTracerSpan:
    """Tests for tracer span functionality."""
    
    def test_span_nesting(self, capsys):
        """Test that spans produce proper indentation."""
        from patentdraw.tracer import get_tracer, configure_tracer
        
        configure_tracer(enabled=True, level="INFO")
        tracer = get_tracer()
        
        with tracer.span("outer", module="test"):
            with tracer.span("inner", module="test"):
                tracer.event("inside")
        
        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")
        
        # Check that we have start/end for both spans plus the event
        assert len(lines) >= 4
        
        # Disable tracer
        configure_tracer(enabled=False)
    
    def test_tracer_disabled_no_output(self, capsys):
        """Test that disabled tracer produces no output."""
        from patentdraw.tracer import get_tracer, configure_tracer
        
        configure_tracer(enabled=False)
        tracer = get_tracer()
        
        with tracer.span("test", module="test"):
            tracer.event("should not appear")
        
        captured = capsys.readouterr()
        assert captured.err == ""


class TestTraceDecorator:
    """Tests for the @trace decorator."""
    
    def test_decorator_runs_function(self):
        """Test that decorated function executes normally."""
        from patentdraw.tracer import trace, configure_tracer
        
        configure_tracer(enabled=False)
        
        @trace(label="test_func")
        def my_func(x):
            return x * 2
        
        result = my_func(5)
        assert result == 10
    
    def test_decorator_with_exception(self):
        """Test that decorator handles exceptions properly."""
        from patentdraw.tracer import trace, configure_tracer
        
        configure_tracer(enabled=False)
        
        @trace(label="failing_func")
        def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            failing_func()
