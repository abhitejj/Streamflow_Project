"""Check the guided choices against the published data and open their links."""

from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect


def check_guided_examples(page, output, url, check_hydro):
    page.goto(url, wait_until="load")
    data = page.evaluate("window.EXPLORER_DATA")
    weather = data["stations"]["8013"]["1w"]
    baseline = data["stations"]["8013"]["8w"]
    gaps = data["stations"]["4021"]["1w"]
    assert weather["flow"]["M2"]["nse"] > weather["flow"]["M1"]["nse"]
    assert baseline["flow"]["M0"]["nse"] > max(value["nse"] for model, value in baseline["flow"].items() if model != "M0")
    assert gaps["n_masked_zero_weeks"] == 379 and gaps["n_eval"] == 142
    expect(page.locator("#example-weather")).to_contain_text("+0.6037 versus M1 +0.4438")
    expect(page.locator("#example-baseline")).to_contain_text("+0.4813 exceeds M1 +0.4021 and M2 +0.2908")
    expect(page.locator("#example-gaps")).to_contain_text("379 recorded-zero weeks are excluded and 142 weeks are scored")
    page.locator("#guided-examples").screenshot(path=output / "guided-examples-desktop.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    destinations = [("weather", "kantamal", "1w", "explorer"), ("baseline", "kantamal", "8w", "explorer"),
                    ("gaps", "karad", "1w", "forecast-replay")]
    for kind, station, lead, anchor in destinations:
        if kind != "weather":
            page.goto(url, wait_until="load")
        link = page.locator(f"#example-{kind}-link")
        target = link.get_attribute("href")
        assert parse_qs(urlsplit(target).query) == {"station": [station], "lead": [lead], "objective": ["flow"], "severity": ["Q95"]}
        assert urlsplit(target).fragment == anchor
        link.press("Enter")
        expect(page.locator("#station-select")).to_have_value(station)
        expect(page.locator(f'#horizon-seg [value="{lead}"]')).to_have_attribute("aria-checked", "true")
        if kind == "weather":
            expect(page.locator("#comparison-table tbody tr").nth(2).get_by_role("cell").first).to_have_text("+0.604")
        elif kind == "baseline":
            expect(page.locator(".comparison-baseline")).to_have_text("Weekly climatology")
        else:
            check_hydro(page, station, lead)
            expect(page.get_by_role("combobox", name="Target week").locator("option").filter(has_text="not scored").first).to_have_count(1)
    return "Three source-backed examples and native keyboard/share destinations checked"
