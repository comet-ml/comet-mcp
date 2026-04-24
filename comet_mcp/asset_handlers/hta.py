"""HolisticTraceAnalysis handler for *_hta_results.json.gz assets.

Install the optional dependency with: pip install comet-mcp[hta]
"""

import os
import tempfile
from typing import Any, Dict, List

MATCH_PATTERN = "*_hta_results.json.gz"


def handle(asset_content: bytes, asset_name: str) -> Dict[str, Any]:
    """Analyze a PyTorch distributed trace asset and return a concise summary.

    Returns temporal breakdown, communication/compute overlap, potential
    stragglers, and the top GPU kernels by time — enough for an LLM to
    diagnose performance issues without being overwhelmed by raw data.
    """
    try:
        from hta.trace_analysis import TraceAnalysis  # type: ignore[import]
    except ImportError:
        return {
            "asset_name": asset_name,
            "error": (
                "HolisticTraceAnalysis is not installed. "
                "Install it with: pip install comet-mcp[hta]"
            ),
        }

    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        f.write(asset_content)
        tmp_path = f.name

    try:
        analyzer = TraceAnalysis(trace_files={0: tmp_path})
        result: Dict[str, Any] = {"asset_name": asset_name}

        result.update(_temporal_breakdown(analyzer))
        result.update(_comm_comp_overlap(analyzer))
        result.update(_stragglers(analyzer))
        result.update(_top_kernels(analyzer))

        return result
    except Exception as e:
        return {"asset_name": asset_name, "error": f"Analysis failed: {e}"}
    finally:
        os.unlink(tmp_path)


def _temporal_breakdown(analyzer: Any) -> Dict[str, Any]:
    try:
        df = analyzer.get_temporal_breakdown(visualize=False)
        return {
            "num_ranks": len(df),
            "temporal_breakdown": {
                "mean_compute_pctg": _pct(df["compute_time_pctg"].mean()),
                "mean_idle_pctg": _pct(df["idle_time_pctg"].mean()),
                "mean_non_compute_pctg": _pct(df["non_compute_time_pctg"].mean()),
            },
        }
    except Exception as e:
        return {"temporal_breakdown": {"error": str(e)}}


def _comm_comp_overlap(analyzer: Any) -> Dict[str, Any]:
    try:
        df = analyzer.get_comm_comp_overlap()
        pcts: List[float] = df["comp_comm_overlap_pctg"].dropna().tolist()
        if not pcts:
            return {}
        return {
            "comm_comp_overlap": {
                "min": _pct(min(pcts)),
                "max": _pct(max(pcts)),
                "mean": _pct(sum(pcts) / len(pcts)),
            }
        }
    except Exception as e:
        return {"comm_comp_overlap": {"error": str(e)}}


def _stragglers(analyzer: Any) -> Dict[str, Any]:
    try:
        return {"potential_stragglers": analyzer.get_potential_stragglers()}
    except Exception as e:
        return {"potential_stragglers": {"error": str(e)}}


def _top_kernels(analyzer: Any) -> Dict[str, Any]:
    try:
        _type_df, kernel_df = analyzer.get_gpu_kernel_breakdown(
            visualize=False, num_kernels=5
        )
        want = [c for c in ["name", "sum", "percentage", "kernel_type"] if c in kernel_df.columns]
        return {"top_kernels_by_time": kernel_df[want].head(5).to_dict(orient="records")}
    except Exception as e:
        return {"top_kernels_by_time": {"error": str(e)}}


def _pct(value: Any) -> float:
    return round(float(value), 3)
