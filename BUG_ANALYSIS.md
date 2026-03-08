# Project Bug Analysis & Fixes

## Critical Bugs Found

### 1. Visualization Method Name Mismatches
**Location:** main.py `_generate_reports()` method (line ~372)

**Issue:**
- Calling `viz_engines["report"].generate()` but class has `generate_strategy_report()`
- Calling `viz_engines["equity_curves"].plot()` but class has `plot_equity_curve()`  
- Calling `viz_engines["trade_signals"].plot()` but class has `plot_trade_analysis()`

**Root Cause:** Wrapper methods don't exist in visualization classes

**Solution:** Add wrapper methods to visualization classes OR refactor main.py to use correct method names

---

### 2. Missing MonthlyReturnsHeatmap Integration
**Location:** visualization + main.py

**Issue:** PerformanceDashboard has `plot_monthly_returns_heatmap()` but not integrated

**Solution:** Add to visualization pipeline

---

### 3. Missing DataMerger Import
**Location:** main.py line 14

**Issue:** `DataMerger` is imported but `_merge_sentiment_data()` implementation uses it directly

**Status:** ✓ Import exists, but need to verify it works

---

### 4. Missing DuckDBEngine Import Path
**Location:** main.py line 54

**Issue:** `from database.duckdb.duckdb_engine import DuckDBEngine` - need to verify path

**Solution:** Check actual module path

---

### 5. Report Generation Flow
**Location:** main.py `_generate_reports()`

**Issue:** Current implementation tries to pass metrics directly to `report.generate()` but:
- ReportGenerator expects matplotlib figures, not raw metrics
- Need to generate figures first from EquityCurveVisualizer and TradeSignalVisualizer

**Solution:** 
1. Get figures from visualization engines
2. Pass figure list to `generate_strategy_report()`

---

## Missing Integrations

### 1. PerformanceDashboard Methods
- `plot_metrics_grid()` - not used
- `plot_attribution_pie()` - not used
- `plot_monthly_returns_heatmap()` - not used
- `plot_risk_metrics_timeline()` - not used
- `plot_performance_vs_risk()` - not used
- `plot_rolling_sharpe()` - not used

**Fix Required:**  Add these visualizations to report generation

### 2. Storage/Persistence Not Fully Integrated
- `_persist_results()` exists but may have path issues
- `_build_storage_manager()` exists but DuckDB path might be wrong

### 3. MLFlow Experiment Tracking
- `_log_experiment_metrics()` exists but MLFlowTracker path validation needed

---

## Non-Critical Issues

### 1. Configuration Keys Mismatch
- Config uses `volatility_window` but code expects `vol_window`
- Config uses `trend_window` but code expects `trend_short` 
- Need config.yaml alignment

### 2. Cross-Asset Model Implementation
- `CrossAssetModel` and `SectorModel` imported but not used in `_apply_model_predictions()`
- Current implementation only uses `EnsemblePredictor.combine_predictions()` with AutoGluon

---

## Risk Assessment

| Bug | Severity | Impact | Fixable |
|-----|----------|--------|---------|
| Visualization method names | CRITICAL | Reports won't generate | ✅ YES |
| Figure generation flow | CRITICAL | No visual output | ✅ YES |
| Config key mismatches | MEDIUM | Runtime errors in regime detection | ✅ YES |
| Cross-asset model unused | LOW | Missing ensemble component | ✅ YES |
| Storage path issues | LOW | Data won't persist | ✅ YES |

---

## Fix Priority

1. **IMMEDIATE:** Fix visualization method names and flow
2. **HIGH:** Fix config key mismatches  
3. **MEDIUM:** Add cross-asset model to ensemble
4. **MEDIUM:** Verify storage paths
5. **LOW:** Add missing visualization methods

