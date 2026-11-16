"""Step-2 mechanics only; no inferential eligibility or winner classification."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import uncertainty_tests as bootstrap
from audit_metrics import compute_continuous_metrics


def test_calendar_reindex_keeps_internal_and_endpoint_missing_positions():
    calendar = pd.date_range('2020-01-06', periods=8, freq='7D')
    frame = pd.DataFrame({'issue_date': calendar[[1, 3, 6]], 'y_obs': [2., 4., 7.],
                          'm1_gbt': [1., 3., 6.], 'm2_gbt': [2., 4., 7.],
                          'scored': [True, False, True]})
    y, m1, m2, mask = bootstrap.calendar_arrays(frame, calendar)
    assert len(y) == 8
    np.testing.assert_array_equal(mask, [False, True, False, False, False, False, True, False])
    assert np.isnan(y[[0, 2, 4, 5, 7]]).all()
    assert y[3] == 4. and not mask[3]  # excluded data stay at their calendar position
    assert m1[6] == 6. and m2[6] == 7.


def test_complete_calendar_comes_from_all_input_rows_not_scored_subset():
    dates = pd.date_range('2020-01-06', periods=60, freq='7D')
    frame = pd.DataFrame({'issue_date': dates[[0, 5, 12, 59]], 'scored': [False, True, True, False]})
    np.testing.assert_array_equal(bootstrap.evaluation_calendar(frame), dates)


def test_shared_sampling_matches_independent_audit_arithmetic(monkeypatch):
    y = np.arange(12, dtype=float) ** 1.2
    m1, m2 = y + np.sin(y), y * .8
    m1[::3] = 1000.  # exercise audit's lower NSE clipping
    m2[::4] = -999.  # exercise physical nonnegative prediction clipping
    mask = np.array([True, False, True, True, False, True, True, True, True, False, True, True])
    indices = np.array([[0, 1, 2, 6, 7, 8, 3, 4, 5, 6, 7, 8],
                        [6, 7, 8, 0, 1, 2, 9, 10, 11, 3, 4, 5]])
    monkeypatch.setattr(bootstrap, 'block_indices', lambda *args: indices)
    point, draws = bootstrap.paired_calendar_bootstrap(y, m1, m2, mask, block=3, n_boot=2)
    expected = []
    for row in indices:
        selected = row[mask[row]]
        expected.append(compute_continuous_metrics(y[selected], m2[selected])['NSE']
                        - compute_continuous_metrics(y[selected], m1[selected])['NSE'])
    np.testing.assert_allclose(draws, expected)
    assert point == bootstrap.observed_delta(y, m1, m2, mask)


@pytest.mark.parametrize('n_scored', [34, 52])
def test_short_observed_series_does_not_force_whole_series_identical_sampling(n_scored):
    calendar_weeks = 522
    y = np.linspace(1., 20., calendar_weeks) ** 1.3
    m1, m2 = y + 1., y + np.sin(np.arange(calendar_weeks)) * 4.
    mask = np.zeros(calendar_weeks, dtype=bool)
    mask[np.linspace(10, 510, n_scored).astype(int)] = True
    _, draws = bootstrap.paired_calendar_bootstrap(y, m1, m2, mask, block=52)
    assert len(draws) == 2000
    assert len(np.unique(draws[np.isfinite(draws)])) > 1


@pytest.mark.parametrize('block', [52, 26, 13])
def test_fixed_seed_reproducible_and_block_indices_remain_calendar_contiguous(block):
    a = bootstrap.block_indices(522, np.random.default_rng(42), block, 20)
    b = bootstrap.block_indices(522, np.random.default_rng(42), block, 20)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0 and a.max() < 522
    for start in range(0, 522, block):
        assert np.diff(a[:, start:min(start + block, 522)], axis=1).min() == 1


@pytest.mark.parametrize('calendar_weeks', [34, 52])
def test_genuinely_short_calendar_rejected_without_silent_block_adaptation(calendar_weeks):
    with pytest.raises(ValueError, match='complete evaluation calendar'):
        bootstrap.block_indices(calendar_weeks, np.random.default_rng(42), 52)


def test_constant_empty_and_nonfinite_draws_are_mechanically_invalid():
    y = np.ones((3, 12))
    mask = np.array([[True]*12, [False]*12, [True]*12])
    y[2] = np.nan
    assert np.isnan(bootstrap.bootstrap_delta(y, y, y, mask)).all()
    # Observed point estimation retains the audit's rule, independently of draw arithmetic.
    assert bootstrap.observed_delta(np.ones(12), np.zeros(12), np.ones(12), np.ones(12, dtype=bool)) == 11.


def test_actual_observed_points_match_all_48_audited_cases():
    predictions = pd.read_csv(bootstrap.RES/'test_predictions.csv', parse_dates=['issue_date'])
    audit = pd.read_csv(bootstrap.RES/'audited_metrics_recomputed.csv')
    calendar = bootstrap.evaluation_calendar(predictions)
    assert len(calendar) == 522
    checked = 0
    for (station, horizon), group in predictions.groupby(['station', 'horizon']):
        arrays = bootstrap.calendar_arrays(group, calendar)
        expected = audit[(audit.station == station) & (audit.horizon == horizon)].iloc[0]
        assert int(arrays[3].sum()) == expected.n_eval
        assert bootstrap.observed_delta(*arrays) == pytest.approx(expected.m2_nse - expected.m1_nse, abs=1e-12)
        checked += 1
    assert checked == 48


def test_excluded_finite_values_never_enter_scoring():
    y = np.arange(80, dtype=float) ** 1.2
    m1, m2 = y + 2., y * .95
    mask = np.arange(80) % 3 != 0
    expected_point, expected_draws = bootstrap.paired_calendar_bootstrap(y, m1, m2, mask, n_boot=50)
    y[~mask], m1[~mask], m2[~mask] = 1e15, 1e30, -1e30
    actual_point, actual_draws = bootstrap.paired_calendar_bootstrap(y, m1, m2, mask, n_boot=50)
    assert expected_point == actual_point
    np.testing.assert_array_equal(expected_draws, actual_draws)


def test_duplicate_dates_rejected_and_shuffled_input_deterministic():
    dates = pd.date_range('2020-01-06', periods=80, freq='7D')
    frame = pd.DataFrame({'station': ['synthetic']*80, 'horizon': ['1w']*80,
                          'issue_date': dates, 'y_obs': np.arange(80, dtype=float),
                          'm1_gbt': np.arange(80, dtype=float)+2,
                          'm2_gbt': np.arange(80, dtype=float)*.9, 'scored': [True]*80})
    summary1, draws1 = bootstrap.run(frame, n_boot=30)
    summary2, draws2 = bootstrap.run(frame.sample(frac=1, random_state=7), n_boot=30)
    pd.testing.assert_frame_equal(summary1, summary2)
    for key in draws1:
        np.testing.assert_array_equal(draws1[key], draws2[key])
    with pytest.raises(ValueError, match='Duplicate'):
        bootstrap.calendar_arrays(pd.concat([frame, frame.iloc[:1]]), dates)


def test_cli_rejects_production_output_directory(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['uncertainty_tests.py', '--output-dir', str(bootstrap.RES)])
    with pytest.raises(SystemExit) as error:
        bootstrap.main()
    assert error.value.code == 2


def diagnostic_fixture():
    return pd.DataFrame({'block_calendar_weeks': [52, 26, 13],
                         'finite_draw_fraction': [.95]*3, 'degenerate': [False]*3,
                         'interval_category': ['spans_zero']*3, 'point_dNSE': [.05]*3})


def test_reliability_boundary_is_prespecified_not_adapted():
    diagnostics = diagnostic_fixture()
    assert bootstrap.assess_reliability(diagnostics, 4.)[0] == 'exploratory_interval'
    assert bootstrap.assess_reliability(diagnostics, 3.999)[0] == 'inference_withheld'
    diagnostics.loc[1, 'finite_draw_fraction'] = 1899/2000
    assert 'finite_draw_fraction' in bootstrap.assess_reliability(diagnostics, 4.)[1]


def test_effective_calendar_blocks_measures_concentration_with_gaps():
    mask = np.zeros(522, dtype=bool)
    mask[:34] = True
    assert bootstrap.effective_calendar_blocks(mask) == 1.
    mask[:] = False
    for start in [0, 52, 104, 156]:
        mask[start:start+10] = True
    assert bootstrap.effective_calendar_blocks(mask) == 4.
    assert bootstrap.effective_calendar_blocks(np.zeros(522, dtype=bool)) == 0.


@pytest.mark.parametrize('category', ['positive', 'negative'])
def test_any_disagreeing_sensitivity_direction_suppresses_inference(category):
    diagnostics = diagnostic_fixture()
    diagnostics.loc[2, 'interval_category'] = category
    status, reason, sensitive = bootstrap.assess_reliability(diagnostics, 8.)
    assert status == 'inference_withheld' and sensitive and 'block_sensitive' in reason


def test_degenerate_or_invalid_sensitivity_cannot_be_saved_as_an_interval():
    diagnostics = diagnostic_fixture()
    diagnostics.loc[1, 'degenerate'] = True
    assert 'degenerate' in bootstrap.assess_reliability(diagnostics, 8.)[1]
    diagnostics.loc[:, 'point_dNSE'] = np.nan
    assert 'observed_point_not_computable' in bootstrap.assess_reliability(diagnostics, 8.)[1]
    with pytest.raises(ValueError, match='All three'):
        bootstrap.assess_reliability(diagnostics.iloc[:1], 8.)


def test_zero_width_percentiles_rejected_even_with_nonzero_raw_spread(monkeypatch):
    dates = pd.date_range('2020-01-06', periods=260, freq='7D')
    frame = pd.DataFrame({'station': ['synthetic']*260, 'horizon': ['1w']*260,
                          'issue_date': dates, 'y_obs': np.arange(260, dtype=float),
                          'm1_gbt': np.arange(260, dtype=float),
                          'm2_gbt': np.arange(260, dtype=float), 'scored': [True]*260})
    monkeypatch.setattr(bootstrap, 'paired_calendar_bootstrap', lambda *args, **kwargs: (.0, np.r_[np.zeros(1999), 1.]))
    diagnostics, _ = bootstrap.run(frame)
    assert diagnostics.degenerate.all()
    result = bootstrap.screened_station_summary(diagnostics).iloc[0]
    assert result.status == 'inference_withheld' and pd.isna(result.ci_lo) and pd.isna(result.ci_hi)
    assert result.dNSE == 0.  # suppression never edits the point estimate


def test_serially_correlated_data_with_calendar_gaps_preserve_paired_resampling():
    rng = np.random.default_rng(73)
    signal = np.empty(260)
    signal[0] = 1.
    for index in range(1, len(signal)):
        signal[index] = .9*signal[index-1] + rng.normal(0, .25)
    y = np.exp(signal) + 1.
    scored = np.ones(260, dtype=bool)
    scored[30:65] = False
    scored[120:160] = False
    frame = pd.DataFrame({'station': ['synthetic']*260, 'horizon': ['1w']*260,
                          'issue_date': pd.date_range('2020-01-06', periods=260, freq='7D'),
                          'y_obs': y, 'm1_gbt': y*.9, 'm2_gbt': y+rng.normal(0,.1,260), 'scored': scored})
    complete_summary, complete_draws = bootstrap.run(frame, n_boot=120)
    sparse_summary, sparse_draws = bootstrap.run(frame[scored], n_boot=120)
    pd.testing.assert_frame_equal(complete_summary, sparse_summary)
    for key in complete_draws:
        np.testing.assert_array_equal(complete_draws[key], sparse_draws[key])


def test_portfolio_output_does_not_resurrect_independent_station_intervals():
    audit = pd.DataFrame({'horizon': ['1w']*4, 'm1_nse': [.1,.2,.3,.4], 'm2_nse': [.2,.3,.4,.5]})
    result = bootstrap.portfolio_point_summary(audit).iloc[0]
    assert result.status == 'inference_withheld'
    assert pd.isna(result.boot_ci_lo) and pd.isna(result.boot_ci_hi)
    assert result.median_paired_dNSE == pytest.approx(.1)
