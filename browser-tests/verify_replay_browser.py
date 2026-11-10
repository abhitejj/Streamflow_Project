"""Exercise the saved forecast replay with the existing Playwright page."""

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import expect


ASSETS = Path(__file__).resolve().parents[1] / "site/assets/replay"
HYDRO = '[data-chart="hydro"]'


def check_replay(page, output, check_hydro):
    page.goto(f"{page.url.split('?')[0]}?station=ashti&lead=1w&objective=drought&severity=Q95#forecast-replay", wait_until="load")
    check_hydro(page, "ashti", "1w")
    chart = page.locator(HYDRO)
    week = page.get_by_role("combobox", name="Target week")
    status = page.locator(".replay-week-status")
    comparison = page.locator("#comparison-table").inner_text()
    chunk = json.loads((ASSETS / "ashti-1w.json").read_text())
    rows = [row for row in chunk["rows"] if row[1] <= "2025-12-31"]
    threshold = chunk["thresholds"]["Q95"]
    initial = next(i for i, row in enumerate(rows) if row[2] and row[4] <= threshold)
    expect(week).to_have_value(str(initial))
    expect(status).to_contain_text(f"Issued {rows[initial][0]} · Target week {rows[initial][1]}")
    expect(status).to_contain_text("Frozen Q95 threshold")
    assert chart.evaluate("chart => chart.layout.shapes[0].y0") == threshold
    assert chart.evaluate("chart => chart.data.every(trace => trace.connectgaps === false)")
    assert chart.evaluate("chart => chart.data[0].x.at(-1)") == rows[-1][1]
    labels = ["Hit" if value <= threshold else "Miss" for value in rows[initial][5:]]
    expect(page.locator(".replay-table tbody .replay-outcome")).to_have_text(labels)
    assert chart.evaluate("chart => chart.layout.shapes[1].x0") == rows[initial][1]
    page.locator("#forecast-replay").screenshot(path=output / "drought-replay.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    replay_objective = page.get_by_role("radiogroup", name="Replay objective")
    replay_objective.get_by_role("radio", name="Flow accuracy").click()
    expect(page.locator('#objective-seg [value="flow"]')).to_have_attribute("aria-checked", "true")
    expect(page.locator(".replay-table .replay-outcome").first).to_be_hidden()
    replay_objective.get_by_role("radio", name="Drought warnings").click()
    severity = page.get_by_role("radiogroup", name="Replay severity")
    severity.get_by_role("radio", name="Moderate (Q90)").click()
    expect(page.locator('#severity-seg [value="Q90"]')).to_have_attribute("aria-checked", "true")
    expect(status).to_contain_text("Frozen Q90 threshold")
    assert chart.evaluate("chart => chart.layout.shapes[0].y0") == chunk["thresholds"]["Q90"]
    severity.get_by_role("radio", name="Severe (Q95)").click()
    expect(week).to_have_value(str(initial))

    following = next(i for i in range(initial + 1, len(rows)) if rows[i][2])
    page.get_by_role("button", name="Next scored week").press("Enter")
    expect(week).to_have_value(str(following))
    page.get_by_role("button", name="Previous scored week").click()
    expect(week).to_have_value(str(initial))

    chart.scroll_into_view_if_needed()
    point = chart.evaluate("""(chart, i) => {
        const {xaxis, yaxis} = chart._fullLayout;
        const trace = chart.data[0];
        const rect = chart.querySelector('.svg-container').getBoundingClientRect();
        return {x: rect.left + xaxis._offset + xaxis.d2p(trace.x[i]),
                y: rect.top + yaxis._offset + yaxis.d2p(trace.y[i])};
    }""", following)
    page.mouse.move(point["x"], point["y"])
    page.mouse.click(point["x"], point["y"])
    expect(week).to_have_value(str(following))
    assert chart.evaluate("chart => chart.layout.shapes[1].x0") == rows[following][1]

    page.get_by_role("button", name="Reset date range").click()
    page.wait_for_function("""() => {
        const chart = document.querySelector('[data-chart="hydro"]');
        return chart.layout.xaxis.range[1] === chart.data[0].x.at(-1);
    }""")
    page.get_by_role("button", name="Zoom to selected week").click()
    page.wait_for_function("""() => {
        const chart = document.querySelector('[data-chart="hydro"]');
        return chart.layout.xaxis.range[1] < chart.data[0].x.at(-1);
    }""")
    assert page.locator("#comparison-table").inner_text() == comparison

    stations = page.get_by_role("combobox", name="Hydrograph station")
    stations.select_option("karad")
    check_hydro(page, "karad", "1w")
    masked = json.loads((ASSETS / "karad-1w.json").read_text())["rows"]
    zero = next(i for i, row in enumerate(masked) if row[3])
    check_unscored(page, zero, "recorded zero")
    next_scored = next(i for i in range(zero + 1, len(masked)) if masked[i][2])
    page.get_by_role("button", name="Next scored week").click()
    expect(week).to_have_value(str(next_scored))
    stations.select_option("haralahalli")
    page.get_by_role("radiogroup", name="Hydrograph lead").get_by_role("radio", name="8 weeks").click()
    check_hydro(page, "haralahalli", "8w")
    haralahalli = json.loads((ASSETS / "haralahalli-8w.json").read_text())["rows"]
    missing = next(i for i, row in enumerate(haralahalli) if not row[2] and not row[3] and row[1] <= "2025-12-31")
    check_unscored(page, missing, "observation or forecast was unavailable")
    page.locator(".replay-panel").screenshot(path=output / "unscored-replay.png", animations="disabled", style=".nav { visibility: hidden; }")

    stations.select_option("warunji")
    page.get_by_role("radiogroup", name="Hydrograph lead").get_by_role("radio", name="2 weeks").click()
    check_hydro(page, "warunji", "2w")
    expect(page.locator(".replay-status")).to_contain_text("No scored drought weeks at Q95")
    warunji = json.loads((ASSETS / "warunji-2w.json").read_text())["rows"]
    expect(week).to_have_value(str(next(i for i, row in enumerate(warunji) if row[2])))
    return "Chart click, scored-week keys, date zoom/reset, frozen threshold, unchanged full-block scores, masked and no-event weeks checked"


def check_unscored(page, index, reason):
    page.get_by_role("combobox", name="Target week").select_option(str(index))
    expect(page.locator(".replay-week-status")).to_contain_text("Not scored")
    expect(page.locator(".replay-week-status")).to_contain_text(reason)
    expect(page.locator(".replay-table tbody .replay-outcome")).to_have_text(["—"] * 5)
    assert page.locator(HYDRO).evaluate("(chart, index) => chart.data.every(trace => trace.y[index] === null)", index)


def check_replay_race(page, url, check_hydro):
    pattern = "**/replay/haralahalli-8w.json?*"
    delayed = []
    page.goto(f"{url}/?station=bamni&lead=8w#forecast-replay", wait_until="load")
    check_hydro(page, "bamni", "8w")
    page.route(pattern, lambda route: delayed.append(route))
    page.get_by_role("combobox", name="Hydrograph station").select_option("haralahalli")
    expect(page.locator(".replay-status")).to_contain_text("Loading replay for Haralahalli")
    expect(page.locator(".replay-panel")).to_be_hidden()
    expect(page.locator(".replay-table tbody tr")).to_have_count(0)
    page.get_by_role("combobox", name="Hydrograph station").select_option("bamni")
    check_hydro(page, "bamni", "8w")
    assert len(delayed) == 1
    with page.expect_request_finished(lambda request: "haralahalli-8w.json" in request.url):
        delayed[0].continue_()
    check_hydro(page, "bamni", "8w")
    page.unroute(pattern)
    return "Late Haralahalli response did not replace Bamni; loading hid the previous details"


def check_all_replays(page, check_hydro):
    stations = page.evaluate("window.STATIONS.map(station => station.id)")
    station_select = page.get_by_role("combobox", name="Hydrograph station")
    leads = page.get_by_role("radiogroup", name="Hydrograph lead")
    for station in stations:
        station_select.select_option(station)
        for lead in ("1w", "2w", "4w", "8w"):
            leads.locator(f'[value="{lead}"]').click()
            check_hydro(page, station, lead)
    return len(stations) * 4


def check_replay_error(browser, url, check_hydro):
    page = browser.new_page(reduced_motion="reduce")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?station=ashti&lead=1w#forecast-replay", wait_until="load")
    check_hydro(page, "ashti", "1w")
    pattern = "**/replay/ashti-2w.json?*"
    page.route(pattern, lambda route: route.fulfill(status=503), times=1)
    page.get_by_role("radiogroup", name="Hydrograph lead").get_by_role("radio", name="2 weeks").click()
    expect(page.locator(".replay-status")).to_have_text("Could not load replay for Ashti at 2 weeks.")
    expect(page.locator(".replay-table tbody tr")).to_have_count(0)
    expect(page.locator(HYDRO)).to_be_hidden()
    page.get_by_role("button", name="Retry loading replay").click()
    check_hydro(page, "ashti", "2w")
    assert not errors, errors
    page.close()
    return "An injected HTTP 503 cleared the chart and detail; Retry restored the selected replay"


def check_prefix(page, url, check_hydro):
    prefix = url + "/streamflow-preview"
    route_glob = prefix + "/**"
    requests = []

    def serve_prefix(route):
        target = route.request.url.removeprefix(prefix)
        requests.append(target)
        path = unquote(urlsplit(target).path).lstrip("/") or "index.html"
        route.fulfill(path=ASSETS.parents[1] / path)

    page.route(route_glob, serve_prefix)
    page.goto(prefix + "/?station=kudige&lead=8w#forecast-replay", wait_until="load")
    check_hydro(page, "kudige", "8w")
    fullscreen = page.locator("#fullscreen-replay").get_attribute("href")
    assert fullscreen.startswith(prefix + "/visualizations/")
    page.goto(fullscreen, wait_until="load")
    check_hydro(page, "kudige", "8w")
    assert sum(path.startswith("/assets/replay/kudige-8w.json") for path in requests) == 2
    page.goto(prefix + "/visualizations/interactive_hydrographs_dashboard.html?station=invalid&lead=constructor", wait_until="load")
    check_hydro(page, "karad", "1w")
    page.unroute(route_glob, serve_prefix)
    return "Main and fullscreen loaded the selected replay from a project-prefixed URL"
