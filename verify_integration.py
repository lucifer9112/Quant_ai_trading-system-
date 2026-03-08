#!/usr/bin/env python3
"""
Integration Verification Script
Validates that all 5 phases are correctly integrated into main.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Phase 1 imports
    try:
        from risk_management.kelly_criterion import KellyCriterion
        from risk_management.position_sizer import AdvancedPositionSizer
        from risk_management.portfolio_risk import PortfolioRiskManager
        from metrics_engine.metrics_aggregator import MetricsAggregator
        from metrics_engine.drawdown_analysis import DrawdownAnalyzer
        print("✅ Phase 1 modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Phase 1 import failed: {e}")
        tests_failed += 1
    
    # Phase 2 imports
    try:
        from ml_models.ensemble import EnsemblePredictor
        from ml_models.cross_asset_model import CrossAssetModel
        from ml_models.sector_model import SectorModel
        from ml_models.autogluon.autogluon_predictor import AutoGluonPredictor
        print("✅ Phase 2 modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Phase 2 import failed: {e}")
        tests_failed += 1
    
    # Phase 3 imports
    try:
        from visualization.report_generator import ReportGenerator
        from visualization.equity_curves import EquityCurveVisualizer
        from visualization.trade_signals import TradeSignalVisualizer
        from visualization.performance_dashboard import PerformanceDashboard
        print("✅ Phase 3 modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Phase 3 import failed: {e}")
        tests_failed += 1
    
    # Phase 4 imports
    try:
        from regime_detection.regime_detector import RegimeDetectionEngine
        from regime_detection.adaptive_allocator import RegimeAwareAllocator
        print("✅ Phase 4 modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Phase 4 import failed: {e}")
        tests_failed += 1
    
    # Phase 5 imports
    try:
        from backtesting.execution.order_execution import ExecutionManager
        from backtesting.execution.portfolio_management import (
            PortfolioConstraints,
            PortfolioRebalancer
        )
        print("✅ Phase 5 modules imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Phase 5 import failed: {e}")
        tests_failed += 1
    
    # Main system import
    try:
        from main import QuantTradingSystem
        print("✅ QuantTradingSystem imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ QuantTradingSystem import failed: {e}")
        tests_failed += 1
    
    return tests_passed, tests_failed


def test_builder_methods():
    """Test that all builder methods exist and return correct objects"""
    print("\n" + "=" * 60)
    print("TESTING BUILDER METHODS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from main import QuantTradingSystem
        system = QuantTradingSystem()
        
        # Test Phase 1 builder
        try:
            risk_mgr = system._build_risk_manager()
            assert isinstance(risk_mgr, dict)
            assert "kelly" in risk_mgr
            assert "position_sizer" in risk_mgr
            assert "portfolio_risk" in risk_mgr
            print("✅ _build_risk_manager() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_risk_manager() failed: {e}")
            tests_failed += 1
        
        # Test metrics builder
        try:
            metrics_engine = system._build_metrics_engine()
            # old versions exposed `calculate`; current API uses `calculate_all_metrics`
            assert hasattr(metrics_engine, 'calculate_all_metrics')
            print("✅ _build_metrics_engine() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_metrics_engine() failed: {e}")
            tests_failed += 1
        
        # Test Phase 4 builder
        try:
            regime = system._build_regime_detector()
            assert isinstance(regime, dict)
            assert "detector" in regime
            assert "allocator" in regime
            print("✅ _build_regime_detector() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_regime_detector() failed: {e}")
            tests_failed += 1
        
        # Test Phase 2 builder
        try:
            ensemble = system._build_ml_ensemble()
            assert isinstance(ensemble, dict)
            assert "ensemble" in ensemble
            print("✅ _build_ml_ensemble() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_ml_ensemble() failed: {e}")
            tests_failed += 1
        
        # Test Phase 3 builder
        try:
            viz = system._build_visualization_engine()
            assert isinstance(viz, dict)
            assert "report" in viz
            assert "equity_curves" in viz
            print("✅ _build_visualization_engine() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_visualization_engine() failed: {e}")
            tests_failed += 1
        
        # Test backtester
        try:
            backtester = system._build_advanced_backtester()
            assert hasattr(backtester, 'backtest')
            print("✅ _build_advanced_backtester() works correctly")
            tests_passed += 1
        except Exception as e:
            print(f"❌ _build_advanced_backtester() failed: {e}")
            tests_failed += 1
        
    except Exception as e:
        print(f"❌ Builder method tests failed: {e}")
        tests_failed += 6
    
    return tests_passed, tests_failed


def test_integration_methods():
    """Test that integration methods exist with correct signatures"""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION METHODS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from main import QuantTradingSystem
        system = QuantTradingSystem()
        
        # Check that new methods exist
        methods_to_check = [
            "_build_risk_manager",
            "_build_metrics_engine",
            "_build_regime_detector",
            "_build_ml_ensemble",
            "_build_visualization_engine",
            "_generate_reports",
            "_apply_regime_aware_allocation",
            "_apply_model_predictions",
        ]
        
        for method_name in methods_to_check:
            if hasattr(system, method_name):
                print(f"✅ Method {method_name} exists")
                tests_passed += 1
            else:
                print(f"❌ Method {method_name} missing")
                tests_failed += 1
    
    except Exception as e:
        print(f"❌ Integration method tests failed: {e}")
        tests_failed = len(methods_to_check)
    
    return tests_passed, tests_failed


def test_configuration():
    """Test that configuration file contains all required sections"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader("config.yaml").load()
        
        required_sections = [
            "trading",
            "backtesting",
            "risk_management",
            "ml_models",
            "visualization",
            "regime_detection",
            "metrics",
        ]
        
        for section in required_sections:
            if section in config:
                print(f"✅ Config section '{section}' present")
                tests_passed += 1
            else:
                print(f"❌ Config section '{section}' missing")
                tests_failed += 1
    
    except Exception as e:
        print(f"❌ Configuration tests failed: {e}")
        tests_failed = len(required_sections)
    
    return tests_passed, tests_failed


def test_source_code():
    """Test that main.py contains all phase integrations"""
    print("\n" + "=" * 60)
    print("TESTING SOURCE CODE")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        main_file = project_root / "main.py"
        
        if not main_file.exists():
            print("❌ main.py not found")
            return 0, 1
        
        with open(main_file) as f:
            main_content = f.read()
        
        # Check for phase imports
        phase_imports = [
            ("Phase 1", "from risk_management.kelly_criterion import KellyCriterion"),
            ("Phase 2", "from ml_models.ensemble import EnsemblePredictor"),
            ("Phase 3", "from visualization.report_generator import ReportGenerator"),
            ("Phase 4", "from regime_detection.regime_detector import RegimeDetectionEngine"),
            ("Phase 5", "from backtesting.execution.order_execution import ExecutionManager"),
        ]
        
        for phase, import_str in phase_imports:
            if import_str in main_content:
                print(f"✅ {phase} imports found")
                tests_passed += 1
            else:
                print(f"❌ {phase} imports missing")
                tests_failed += 1
        
        # Check for phase integration methods
        phase_methods = [
            "_build_risk_manager",
            "_build_metrics_engine",
            "_build_regime_detector",
            "_build_ml_ensemble",
            "_build_visualization_engine",
            "_generate_reports",
            "_apply_regime_aware_allocation",
        ]
        
        for method in phase_methods:
            if f"def {method}" in main_content:
                print(f"✅ Method {method} defined")
                tests_passed += 1
            else:
                print(f"❌ Method {method} not defined")
                tests_failed += 1
    
    except Exception as e:
        print(f"❌ Source code tests failed: {e}")
        tests_failed += 12
    
    return tests_passed, tests_failed


def main():
    """Run all integration tests"""
    print("\n")
    # ascii-friendly header avoids unicode encoding issues on Windows
    print("+" + "=" * 58 + "+")
    print("|" + " " * 58 + "|")
    print("|" + "INTEGRATION VERIFICATION SUITE".center(58) + "|")
    print("|" + "All 5 Phases Integration Test".center(58) + "|")
    print("|" + " " * 58 + "|")
    print("+" + "=" * 58 + "+")
    
    total_passed = 0
    total_failed = 0
    
    # Run test suites
    p, f = test_imports()
    total_passed += p
    total_failed += f
    
    p, f = test_builder_methods()
    total_passed += p
    total_failed += f
    
    p, f = test_integration_methods()
    total_passed += p
    total_failed += f
    
    p, f = test_configuration()
    total_passed += p
    total_failed += f
    
    p, f = test_source_code()
    total_passed += p
    total_failed += f
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {100 * total_passed / (total_passed + total_failed):.1f}%")
    
    if total_failed == 0:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("System is ready for backtesting.")
        return 0
    else:
        print(f"\n❌ {total_failed} test(s) failed.")
        print("Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
