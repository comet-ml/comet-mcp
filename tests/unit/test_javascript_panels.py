#!/usr/bin/env python3
"""
Unit tests for the dashboard view and JavaScript panel tools.
"""

from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

views = pytest.importorskip("comet_ml.views")
if not hasattr(views, "javascript_panel"):
    pytest.skip("comet_ml.views has no javascript_panel", allow_module_level=True)

from comet_mcp.tools import (
    create_javascript_panel,
    get_javascript_panel,
    list_views,
    update_javascript_panel,
)
from comet_mcp.utils import ToolRegistry


def _saved(view):
    """What the backend hands back: the view, now with an id."""
    view.template_id = view.template_id or "new-view"
    return view


def _dashboard() -> "views.View":
    js = views.javascript_panel("class P extends Comet.Panel {}", name="Mine")
    js["chartId"] = js["storeId"] = "js1"
    line = views.line_panel("loss", name="Loss")
    line["chartId"] = "line1"
    view = views.build_view(
        "Main", sections=[views.Section("Metrics", [line, js], columns=2)]
    )
    view.template_id = "view1"
    return view


@pytest.fixture
def api():
    api = Mock()
    api.get_views.return_value = [_dashboard()]
    api.create_view.side_effect = lambda ws, project, view: _saved(view)
    api.update_view.side_effect = lambda ws, project, view: _saved(view)
    with patch("comet_mcp.tools.get_comet_api") as get_api:
        get_api.return_value.__enter__.return_value = api
        yield api


class TestListViews:
    def test_list_views__sections_and_panels(self, api):
        result = list_views("proj", workspace="ws")

        assert result["views"] == [
            {
                "view_id": "view1",
                "name": "Main",
                "sections": [
                    {
                        "title": "Metrics",
                        "panels": [
                            {"chart_id": "line1", "type": "line", "name": "Loss"},
                            {"chart_id": "js1", "type": "javascript", "name": "Mine"},
                        ],
                    }
                ],
            }
        ]
        api.get_views.assert_called_once_with("ws", "proj")


class TestCreateJavascriptPanel:
    def test_create__no_view_id__creates_view(self, api):
        result = create_javascript_panel(
            "proj", "code()", panel_name="Mine", workspace="ws"
        )

        view = api.create_view.call_args.args[2]
        panel = view.v3["sections"][0]["panels"][0]
        assert panel["chartType"] == "javascript"
        assert panel["code"]["code"] == "code()"
        assert view.name == "Mine"
        assert result["status"] == "created"
        assert result["view_id"] == "new-view"
        assert result["chart_id"] == panel["chartId"]
        assert result["section"] == "JavaScript panels"
        api.update_view.assert_not_called()

    def test_create__view_id__appended_below_first_section(self, api):
        result = create_javascript_panel(
            "proj",
            "code()",
            view_id="view1",
            resources=[{"url": "https://x/a.css", "type": "stylesheet"}],
            workspace="ws",
        )

        view = api.update_view.call_args.args[2]
        section = view.v3["sections"][0]
        assert view.template_id == "view1"
        assert [p["chartId"] for p in section["panels"]][-1] == result["chart_id"]
        assert section["panels"][-1]["code"]["resources"] == [
            {"url": "https://x/a.css", "type": "stylesheet"}
        ]
        # Two panels side by side fill row 0, so the new one starts row 1.
        assert section["layout"][-1]["i"] == result["chart_id"]
        assert section["layout"][-1]["y"] == 1
        api.create_view.assert_not_called()

    def test_create__new_section_title__section_added(self, api):
        result = create_javascript_panel(
            "proj", "code()", view_id="view1", section_title="Custom", workspace="ws"
        )

        view = api.update_view.call_args.args[2]
        assert [s["title"] for s in view.v3["sections"]] == ["Metrics", "Custom"]
        assert view.v3["sections"][1]["layout"][0]["y"] == 0
        assert result["section"] == "Custom"

    def test_create__unknown_view__raises(self, api):
        with pytest.raises(Exception, match="not found"):
            create_javascript_panel("proj", "code()", view_id="nope", workspace="ws")

    def test_create__bad_resource_type__raises(self, api):
        with pytest.raises(ValueError):
            create_javascript_panel(
                "proj",
                "code()",
                resources=[{"url": "https://x/a.js", "type": "module"}],
                workspace="ws",
            )


class TestGetJavascriptPanel:
    def test_get__javascript_panel__code_returned(self, api):
        result = get_javascript_panel("proj", "view1", "js1", workspace="ws")

        assert result["name"] == "Mine"
        assert result["code"] == "class P extends Comet.Panel {}"
        assert result["html"] == views.JS_PANEL_DEFAULT_HTML
        assert result["resources"] == []

    def test_get__other_panel_type__raises(self, api):
        with pytest.raises(Exception, match="not a JavaScript panel"):
            get_javascript_panel("proj", "view1", "line1", workspace="ws")


class TestUpdateJavascriptPanel:
    def test_update__code_only__rest_kept(self, api):
        result = update_javascript_panel(
            "proj", "view1", "js1", code="new()", workspace="ws"
        )

        view = api.update_view.call_args.args[2]
        panel = view.v3["sections"][0]["panels"][1]
        assert panel["chartId"] == "js1"
        assert panel["storeId"] == "js1"
        assert panel["chartName"] == "Mine"
        assert panel["code"]["code"] == "new()"
        assert panel["code"]["html"] == views.JS_PANEL_DEFAULT_HTML
        assert result["changed"] == ["code"]

    def test_update__name__renamed(self, api):
        update_javascript_panel(
            "proj", "view1", "js1", panel_name="Renamed", workspace="ws"
        )

        view = api.update_view.call_args.args[2]
        assert view.v3["sections"][0]["panels"][1]["chartName"] == "Renamed"

    def test_update__unknown_panel__raises(self, api):
        with pytest.raises(Exception, match="not found"):
            update_javascript_panel("proj", "view1", "nope", code="x", workspace="ws")


class TestArraySchemas:
    def test_optional_list__items_are_the_list_items(self):
        def tool(
            names: Optional[List[str]] = None,
            resources: Optional[List[Dict[str, str]]] = None,
        ) -> None:
            """A tool."""

        properties = ToolRegistry()._generate_input_schema(tool)["properties"]

        assert properties["names"]["items"] == {"type": "string"}
        assert properties["resources"]["items"] == {"type": "object"}
