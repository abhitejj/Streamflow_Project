"""Run the ES 335 site smoke test against http://localhost:8000 with Chromium."""

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright
from verify_explorer_data import verify_explorer_data
from verify_replay_browser import check_all_replays, check_prefix, check_replay, check_replay_error, check_replay_race
from verify_replay_data import verify_replay_data


HERE = Path(__file__).resolve().parent
SITE = HERE.parent / "site"
URL = "http://localhost:8000"
CHARTS = ("portfolio", "skill", "hydro", "pr", "transfer", "kge", "fi")
HYDRO = '[data-chart="hydro"]'
TRANSFER = '[data-chart="transfer"]'
MODELS = ("M0", "M1", "M2", "M1-log", "M2-log")


def screenshot_explorer(page, path):
    page.locator(".explorer").screenshot(path=path, animations="disabled", style=".nav { visibility: hidden; }")


def listen_for_errors(page, report):
    page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
    page.on("console", lambda message: report["console_errors"].append(message.text)
            if message.type == "error" else None)
    page.on("requestfailed", lambda request: report["network_failures"].append(
        f"{request.url}: {request.failure}"
    ))
    page.on("response", lambda response: report["network_failures"].append(
        f"{response.status} {response.url}"
    ) if response.status >= 400 else None)


def check_chart(page, selector):
    chart = page.locator(selector)
    chart.scroll_into_view_if_needed()
    expect(chart.locator(".scatterlayer").first).to_be_visible(timeout=10000)
    trace_count = chart.evaluate("element => element._fullData.length")
    assert trace_count > 0, f"{selector} rendered without traces"
    return trace_count


def check_heatmap(page):
    cells = page.locator("#heat .cell")
    expect(cells).to_have_count(48)
    mismatches = cells.evaluate_all("""cells => {
        const stations = new Map(window.STATIONS.map(station => [station.id, station]));
        return cells.flatMap(cell => {
            const {id, h} = cell.dataset;
            const expected = stations.get(id).metrics[h].delta_nse_m2_m1;
            const actual = cell.textContent.replace('−', '-');
            return actual === expected ? [] : [{id, h, actual, expected}];
        });
    }""")
    assert not mismatches, mismatches
    return cells.count()


def check_worst_score(page, selector, axis, worst_score):
    page.wait_for_function("""({selector, axis, worstScore}) => {
        const chart = document.querySelector(selector);
        const range = chart?._fullLayout?.[axis + 'axis']?.range;
        const plotted = chart?.data?.flatMap(trace => trace[axis]);
        return range && plotted?.includes(worstScore) &&
            plotted.filter(Number.isFinite).every(value => range[0] < value && value < range[1]);
    }""", arg={"selector": selector, "axis": axis, "worstScore": worst_score}, timeout=5000)
    return {"worst_score": worst_score, "range": page.locator(selector).evaluate(
        "(chart, axis) => chart._fullLayout[axis + 'axis'].range", axis
    )}


def check_explorer(page):
    station_select = page.get_by_label("Gauging Station Sub-Basin")
    station_select.select_option("kantamal")
    lead = page.locator('#horizon-seg [value="4w"]')
    lead.click()
    expect(station_select).to_have_value("kantamal")
    expect(lead).to_have_attribute("aria-checked", "true")
    expect(page.locator('#heat .cell[data-id="kantamal"][data-h="4w"].sel')).to_have_count(1)

    check_comparison(page, "kantamal", "4w", "flow")
    expect(page.locator(".comparison-baseline")).to_have_text("Weekly climatology")
    page.locator('#heat .cell[data-id="karad"][data-h="2w"]').click()
    expect(station_select).to_have_value("karad")
    expect(page.locator('#horizon-seg [value="2w"]')).to_have_attribute("aria-checked", "true")
    expect(page.locator(".comparison-baseline")).to_have_text("Persistence")
    return {"dropdown_and_lead": "kantamal / 4w", "heatmap_click": "karad / 2w"}


def check_selection(page, station, lead):
    expect(page.locator("#station-select")).to_have_value(station)
    expect(page.locator(f'#horizon-seg [value="{lead}"]')).to_have_attribute("aria-checked", "true")
    expect(page.locator(f'#heat [data-id="{station}"][data-h="{lead}"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#selection-status")).to_contain_text(station.title())
    expect(page.locator("#selection-status")).to_contain_text(f"{lead[0]} week")


def check_hydro(page, station, lead):
    hydro = page.locator(HYDRO)
    body = hydro.locator("..")
    body.scroll_into_view_if_needed()
    expect(body.get_by_role("combobox", name="Hydrograph station")).to_have_value(station)
    expect(body.locator(f'[aria-label="Hydrograph lead"] [value="{lead}"]')).to_have_attribute("aria-checked", "true")
    expect(hydro).to_be_visible()
    expect(hydro).to_have_attribute("aria-label", f"Observed and forecast discharge for {station.title()} at {lead[0]} {'week' if lead == '1w' else 'weeks'}")
    expect(body.get_by_role("combobox", name="Target week")).to_be_visible()
    assert hydro.evaluate("chart => chart.data.length") == 6


def check_linked_controls(page):
    check_hydro(page, "karad", "2w")
    page.get_by_role("combobox", name="Hydrograph station").select_option("kantamal")
    hydro_leads = page.get_by_role("radiogroup", name="Hydrograph lead")
    hydro_leads.get_by_role("radio", name="2 weeks").press("ArrowRight")
    check_selection(page, "kantamal", "4w")
    check_hydro(page, "kantamal", "4w")
    expect(hydro_leads.get_by_role("radio", name="4 weeks")).to_be_focused()
    expect(page.get_by_role("radiogroup", name="Transfer lead").get_by_role("radio", name="4 weeks")).to_have_attribute("aria-checked", "true")
    assert page.locator("#station-mini").evaluate("chart => chart.data[2].marker.size") == [8, 8, 14, 8]
    share_url = page.locator("#share-selection").get_attribute("href")
    assert parse_qs(urlsplit(share_url).query) == {"station": ["kantamal"], "lead": ["4w"], "objective": ["flow"], "severity": ["Q95"]}
    assert urlsplit(share_url).fragment == "explorer"

    point = page.locator("#station-mini .scatterlayer .trace").last.locator(".point").first
    point.scroll_into_view_if_needed()
    bounds = point.bounding_box()
    page.mouse.click(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
    check_selection(page, "kantamal", "1w")
    assert page.locator("#station-mini").evaluate("chart => chart.data[2].marker.size") == [14, 8, 8, 8]
    page.go_back()
    check_selection(page, "kantamal", "4w")
    page.go_forward()
    check_selection(page, "kantamal", "1w")

    leads = page.locator("#horizon-seg")
    for before, key, after in [("1w", "End", "8w"), ("8w", "ArrowLeft", "4w"), ("4w", "Home", "1w")]:
        leads.locator(f'[value="{before}"]').press(key)
        check_selection(page, "kantamal", after)
        expect(leads.locator(f'[value="{after}"]')).to_be_focused()
    leads.locator('[value="2w"]').press("Enter")
    check_selection(page, "kantamal", "2w")
    leads.locator('[value="4w"]').press("Space")
    check_selection(page, "kantamal", "4w")
    expect(leads.locator('[tabindex="0"]')).to_have_count(1)

    pr_leads = page.get_by_role("radiogroup", name="Pooled PR lead")
    pr_leads.get_by_role("radio", name="4 weeks").click()
    check_selection(page, "kantamal", "4w")
    pr_leads.get_by_role("radio", name="2 weeks").click()
    check_selection(page, "kantamal", "4w")
    return share_url


def check_direct_navigation(page, share_url):
    page.goto(share_url, wait_until="load")
    check_selection(page, "kantamal", "4w")
    check_hydro(page, "kantamal", "4w")
    page.locator('.nav-links a[href="#methodology"]').click()
    assert urlsplit(page.url).fragment == "methodology"
    check_selection(page, "kantamal", "4w")
    page.goto(f"{URL}/?station=missing&lead=99w&objective=bogus&severity=Q50#explorer", wait_until="load")
    check_selection(page, "karad", "1w")
    check_hydro(page, "karad", "1w")
    check_comparison(page, "karad", "1w", "flow")


def get_entry(page, station, lead):
    return page.evaluate("""({station, lead}) => {
        const gauge = window.STATIONS.find(s => s.id === station).camels_id;
        return window.EXPLORER_DATA.stations[gauge][lead];
    }""", {"station": station, "lead": lead})


def check_comparison(page, station, lead, mode, severity="Q95"):
    entry = get_entry(page, station, lead)
    table = page.locator("#comparison-table")
    expect(table.locator("tbody tr")).to_have_count(5)
    expect(page.locator(f'#objective-seg [value="{mode}"]')).to_have_attribute("aria-checked", "true")
    context = page.locator("#comparison-context")
    expect(context).to_contain_text(f'{entry["n_eval"]} scored weeks')
    if mode == "drought":
        group = entry["drought"][severity]
        expect(page.locator(f'#severity-seg [value="{severity}"]')).to_have_attribute("aria-checked", "true")
        expect(context).to_contain_text(f'{group["events"]} observed drought weeks')
        expect(context).to_contain_text("primary" if severity == "Q95" else "sensitivity")
        values = group["models"]
        fields = ("hits", "misses", "false_alarms", "csi", "pod", "far")
        metric = f"{severity} CSI"
    else:
        expect(page.locator("#severity-field")).to_be_hidden()
        expect(page.locator("#comparison-note")).to_contain_text("Log-NSE scores all models on log(1 + discharge)")
        values = entry["flow"]
        fields = ("nse", "log_nse", "kge")
        metric = "NSE"
    for index, model in enumerate(MODELS):
        row = table.locator("tbody tr").nth(index)
        expect(row.get_by_role("rowheader")).to_contain_text(model)
        expected = []
        for field in fields:
            value = values[model][field]
            if value is None:
                expected.append("undefined")
            elif mode == "flow":
                expected.append(f"{value:+.3f}".replace("-", "−"))
            else:
                expected.append(str(value) if field in fields[:3] else f"{value:.3f}")
        expect(row.get_by_role("cell")).to_have_text(expected)
    expect(page.locator("#station-mini")).to_have_attribute("aria-label", f"{metric} by lead time for {station.title()}. Selected lead: {lead[0]} {'week' if lead == '1w' else 'weeks'}.")
    assert page.locator("#station-mini").evaluate("chart => chart.data.map(trace => trace.name)") == list(MODELS)
    return entry


def check_objectives(page, output):
    heat = page.locator("#heat")
    heat.locator('[data-id="warunji"][data-h="2w"]').click()
    page.locator('#objective-seg [value="flow"]').press("ArrowRight")
    check_selection(page, "warunji", "2w")
    check_comparison(page, "warunji", "2w", "drought")
    expect(page.locator("#comparison-context")).to_contain_text("0 observed drought weeks")
    expect(page.locator("#comparison-table tbody tr").first.get_by_role("cell")).to_have_text(
        ["0", "0", "0", "undefined", "undefined", "undefined"]
    )

    heat.locator('[data-id="karad"][data-h="1w"]').click()
    check_comparison(page, "karad", "1w", "drought")
    expect(page.locator("#comparison-table tbody tr").nth(2).get_by_role("cell")).to_have_text(
        ["0", "1", "0", "0.000", "0.000", "undefined"]
    )

    heat.locator('[data-id="ashti"][data-h="1w"]').click()
    page.locator('#severity-seg [value="Q95"]').press("ArrowRight")
    check_comparison(page, "ashti", "1w", "drought", "Q90")
    page.locator('#severity-seg [value="Q90"]').press("End")
    entry = check_comparison(page, "ashti", "1w", "drought", "Q80")
    share_url = page.locator("#share-selection").get_attribute("href")
    assert parse_qs(urlsplit(share_url).query) == {"station": ["ashti"], "lead": ["1w"], "objective": ["drought"], "severity": ["Q80"]}
    with page.expect_download() as download_event:
        page.get_by_role("link", name="Download this comparison (CSV)").click()
    download = download_event.value
    assert download.suggested_filename == "ashti-1w-Q80.csv"
    download_path = output / download.suggested_filename
    download.save_as(download_path)
    with download_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["model"] for row in rows] == list(MODELS)
    source = page.evaluate("window.EXPLORER_DATA.sources.drought")
    assert all(float(row["threshold_m3_s"]) == entry["drought"]["Q80"]["threshold"] and
               row["source_sha256"] == source["sha256"] and row["source"] == source["path"] for row in rows)
    screenshot_explorer(page, output / "drought-comparison.png")

    page.locator('#objective-seg [value="flow"]').click()
    check_comparison(page, "ashti", "1w", "flow")
    screenshot_explorer(page, output / "flow-comparison.png")
    page.go_back()
    check_comparison(page, "ashti", "1w", "drought", "Q80")
    page.go_forward()
    check_comparison(page, "ashti", "1w", "flow")
    return share_url


def check_standalone(page):
    for file, chart in [("interactive_hydrographs_dashboard", "hydro"),
                        ("interactive_spatial_transfer_grouped_bars", "transfer")]:
        page.goto(f"{URL}/visualizations/{file}.html", wait_until="load")
        assert not page.evaluate("Boolean(document.querySelector('script[src$=\"/site.js\"]'))")
        check_chart(page, f'[data-chart="{chart}"]')
        page.get_by_role("radio", name="4 weeks" if chart == "hydro" else "8 weeks").click()
        if chart == "transfer":
            check_worst_score(page, TRANSFER, "x", -1.942)
        else:
            expect(page.locator(HYDRO)).to_have_attribute("aria-label", "Observed and forecast discharge for Karad at 4 weeks")
            expect(page.get_by_role("combobox", name="Target week")).to_be_visible()
    for station in ("karad", "kantamal"):
        for lead in ("1w", "4w"):
            page.goto(f"{URL}/visualizations/interactive_hydrograph_{station}_h{lead}.html", wait_until="load")
            check_chart(page, f'[data-chart="hydro:{station}:{lead}"]')
            expect(page.get_by_role("combobox", name="Target week")).to_be_visible()
            expect(page.get_by_role("combobox", name="Hydrograph station")).to_have_count(0)


def check_mobile(page, output, share_url):
    page.goto(share_url, wait_until="load")
    check_selection(page, "kantamal", "4w")
    for name in CHARTS:
        check_chart(page, f'[data-chart="{name}"]')
    check_chart(page, "#station-mini")
    page.locator('#heat .cell[data-id="warunji"][data-h="2w"]').tap()
    check_selection(page, "warunji", "2w")
    check_hydro(page, "warunji", "2w")
    widths = page.evaluate("""() => ({
        viewport: document.documentElement.clientWidth,
        document: document.documentElement.scrollWidth,
        body: document.body.scrollWidth
    })""")
    assert max(widths["document"], widths["body"]) <= widths["viewport"], widths
    screenshot_explorer(page, output / "mobile-station-explorer.png")
    return widths


def run(output):
    report = {"url": URL, "charts": {}, "page_errors": [], "console_errors": [], "network_failures": []}
    output.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright, playwright.chromium.launch() as browser:
            report["chromium"] = browser.version
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            listen_for_errors(page, report)

            response = page.goto(URL, wait_until="load", timeout=30000)
            assert response and response.ok, "Home page did not return HTTP success"
            assert response.body() == (SITE / "index.html").read_bytes(), "localhost:8000 serves a different index.html"
            for name in ("site.js", "charts.js", "site.css", "benchmark-data.js", "explorer-data.js", "replay-manifest.js", "figdata.js"):
                asset = page.request.get(f"{URL}/assets/{name}")
                assert asset.ok and asset.body() == (SITE / "assets" / name).read_bytes(), f"Served {name} differs from site/"
            report["served_worktree"] = str(SITE)

            for name in CHARTS:
                report["charts"][name] = check_chart(page, f'[data-chart="{name}"]')
            report["charts"]["station-mini"] = check_chart(page, "#station-mini")
            report["explorer"] = check_explorer(page)
            share_url = check_linked_controls(page)
            report["linked_controls"] = "Hydrograph, mini chart, keyboard, transfer, pooled PR, and browser history checked"
            report["audited_heatmap_cells"] = check_heatmap(page)
            page.locator('#heat .cell[data-id="warunji"][data-h="2w"]').click()
            warunji_score = get_entry(page, "warunji", "2w")["flow"]["M0"]["nse"]
            report["station_axis"] = check_worst_score(page, "#station-mini", "y", warunji_score)
            screenshot_explorer(page, output / "station-explorer.png")
            transfer = page.locator(TRANSFER)
            transfer.locator("..").get_by_role("radio", name="8 weeks").click()
            check_selection(page, "warunji", "8w")
            report["transfer_axis"] = check_worst_score(page, TRANSFER, "x", -1.942)
            transfer.screenshot(path=output / "spatial-transfer.png", animations="disabled")
            page.get_by_role("combobox", name="Hydrograph station").select_option("kantamal")
            page.get_by_role("radiogroup", name="Hydrograph lead").get_by_role("radio", name="4 weeks").click()
            check_hydro(page, "kantamal", "4w")
            page.locator(HYDRO).locator("..").screenshot(path=output / "linked-hydrograph.png", animations="disabled")
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=output / "full-page.png", full_page=True, animations="disabled")
            drought_url = check_objectives(page, output)
            report["comparisons"] = verify_explorer_data()
            report["comparison_interactions"] = "Flow, three severities, undefined ratios, CSV download, and history checked"

            other = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            listen_for_errors(other, report)
            check_direct_navigation(other, share_url)
            other.goto(drought_url, wait_until="load")
            check_selection(other, "ashti", "1w")
            check_comparison(other, "ashti", "1w", "drought", "Q80")
            report["direct_links"] = "Fresh load, lazy hydrograph, section anchor, and invalid fallback checked"
            report["replay_data"] = verify_replay_data()
            report["replay_interactions"] = check_replay(other, output, check_hydro)
            report["replay_race"] = check_replay_race(other, URL, check_hydro)
            report["replay_browser_selections"] = check_all_replays(other, check_hydro)
            check_standalone(other)
            report["standalone"] = ["hydro", "four fixed hydrographs", "transfer"]
            report["project_prefix"] = check_prefix(other, URL, check_hydro)
            report["replay_retry"] = check_replay_error(browser, URL, check_hydro)
            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, reduced_motion="reduce")
            listen_for_errors(mobile, report)
            report["mobile_widths"] = check_mobile(mobile, output, share_url)
            mobile.goto(drought_url, wait_until="load")
            check_comparison(mobile, "ashti", "1w", "drought", "Q80")
            scroller = mobile.locator(".comparison-scroll")
            assert scroller.evaluate("region => region.scrollWidth > region.clientWidth")
            scroller.evaluate("region => region.scrollLeft = region.scrollWidth")
            assert scroller.evaluate("region => region.scrollLeft > 0")
            assert mobile.evaluate("document.documentElement.scrollWidth === 390 && document.body.scrollWidth === 390")
            screenshot_explorer(mobile, output / "mobile-drought-comparison.png")
            check_hydro(mobile, "ashti", "1w")
            mobile.get_by_role("button", name="Next scored week").tap()
            expect(mobile.locator(".replay-week-status")).to_contain_text("Scored in the full test block")
            mobile.locator("#forecast-replay").screenshot(path=output / "mobile-replay.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
            assert not report["page_errors"], report["page_errors"]
            assert not report["console_errors"], report["console_errors"]
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts")
    run(parser.parse_args().output)
