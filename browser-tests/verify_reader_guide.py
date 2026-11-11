"""Check the reader guide, reported-test caveats, and static fallback."""

from playwright.sync_api import expect


def check_reader_guide(page, output):
    expect(page.locator("h1")).to_have_text("Streamflow & Hydrological Drought Forecasting")
    expect(page.locator(".hero-takeaway")).to_have_text("Weather inputs improve short-lead NSE over flow-only models.")
    guide = page.locator("#metric-guide")
    guide.get_by_text("Metric guide · definitions and score direction", exact=True).press("Enter")
    expect(guide.locator("details")).to_have_attribute("open", "")
    expect(guide.locator("dt")).to_have_count(9)
    expect(guide).to_contain_text("Negative values are worse than that reference")
    expect(guide).to_contain_text("Zero is not a universal benchmark")
    expect(guide).to_contain_text("differs from the false positive rate")
    expect(guide).to_contain_text("average precision")
    expect(guide).to_contain_text("training-period 20th, 10th, and 5th percentiles")
    guide.screenshot(path=output / "metric-guide-desktop.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    notes = page.locator("#comparison-notes")
    notes.get_by_text("Test interpretation, baselines, and sample sizes", exact=True).press("Enter")
    rows = notes.locator("tbody tr")
    medians = page.evaluate("window.MEDIANS")
    for index, lead in enumerate(("1w", "2w", "4w", "8w")):
        expect(rows.nth(index).get_by_role("cell")).to_have_text([
            medians[lead]["p_nse_m2_m1"], medians[lead]["p_kge_m2_m1"]
        ])
    expect(notes).to_contain_text("0.0125")
    expect(notes).to_contain_text("nested gauges")
    expect(notes).to_contain_text("different station subsets")
    notes.screenshot(path=output / "comparison-notes.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    page.locator(".hero").screenshot(path=output / "hero-desktop.png", animations="disabled", style=".nav, .to-top { visibility: hidden; }")
    return "Native keyboard guide, nine metric definitions, unchanged reported p-values, and comparison caveats checked"


def check_static_fallback(browser, url, output):
    page = browser.new_page(java_script_enabled=False, viewport={"width": 1440, "height": 1000})
    page.goto(url, wait_until="load")
    link = page.get_by_role("link", name="static benchmark table", exact=True)
    expect(link).to_be_visible()
    link.click()
    table = page.locator("#benchmark-matrix")
    expect(table.locator("tbody tr")).to_have_count(16)
    expect(table).to_contain_text("+0.045 (p=0.027)")
    page.screenshot(path=output / "static-benchmark.png", animations="disabled")
    page.close()
    return "With JavaScript disabled, the fallback link opens the 16-row static benchmark table"
