"""Check evidence labels, selection, and uncertainty downloads in Chromium."""

import csv
from playwright.sync_api import expect


def check_uncertainty(page, output):
    source = page.evaluate("window.EXPLORER_DATA")
    stations = page.evaluate("window.STATIONS")
    labels = {"M2 better": "M2", "M1 better": "M1", "not distinguishable": "?"}
    toggle = page.get_by_role("radiogroup", name="Heatmap view")
    toggle.get_by_role("radio", name="NSE difference").press("ArrowRight")
    expect(toggle.get_by_role("radio", name="Uncertainty evidence")).to_be_focused()
    expect(page.locator("#evidence-legend")).to_be_visible()
    expect(page.locator("#delta-legend")).to_be_hidden()
    for station in stations:
        for lead, entry in source["stations"][station["camels_id"]].items():
            cell = page.locator(f'#heat [data-id="{station["id"]}"][data-h="{lead}"]')
            expect(cell).to_have_text(labels[entry["uncertainty"]["verdict"]])
            assert "95% interval" in cell.get_attribute("aria-label")
    page.locator('#heat [data-id="warunji"][data-h="1w"]').press("Enter")
    expect(page.locator("#station-select")).to_have_value("warunji")
    expect(page.locator("#station-uncertainty")).to_contain_text("Inconclusive")
    expect(page.locator("#station-uncertainty")).to_contain_text("133 scored weeks")
    expected = page.locator("#station-uncertainty").text_content()
    page.locator('#objective-seg [value="drought"]').click()
    page.locator('#severity-seg [value="Q80"]').click()
    expect(page.locator("#station-uncertainty")).to_have_text(expected)
    expect(page.locator("#portfolio-uncertainty")).to_contain_text("12 stations")
    page.get_by_text("How to read these intervals", exact=True).click()
    expect(page.locator(".uncertainty details")).to_contain_text("closing gaps")
    expect(page.locator(".uncertainty details")).to_contain_text("does not establish equal performance")
    with page.expect_download() as event:
        page.get_by_role("link", name="Download these intervals (CSV)").click()
    download = event.value
    path = output / download.suggested_filename
    download.save_as(path)
    rows = list(csv.DictReader(path.open()))
    assert [row["scope"] for row in rows] == ["station", "portfolio"]
    assert rows[0]["source_sha256"] == source["sources"]["station_uncertainty"]["sha256"]
    assert rows[1]["source_sha256"] == source["sources"]["portfolio_uncertainty"]["sha256"]
    page.locator("#explorer").screenshot(path=output / "uncertainty-desktop.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    toggle.get_by_role("radio", name="Uncertainty evidence").press("Home")
    expect(page.locator('#heat [data-id="warunji"][data-h="1w"]')).to_have_text("+0.030")
    return "All48 evidence cells, keyboard selection, objective independence, method explanation, and download checked"
