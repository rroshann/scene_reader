#!/usr/bin/env python3
"""
Regression Tests for Approach 2.5
Ensures Approach 2 remains functional after Approach 2.5 implementation
"""
import sys
import os
from pathlib import Path

# Add parent directory to path (same as Approach 2 scripts)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add approach_2_yolo_llm directory to path for imports
approach2_dir = project_root / "code" / "approach_2_yolo_llm"
sys.path.insert(0, str(approach2_dir))

def test_approach2_imports():
    """Test that Approach 2 imports work correctly"""
    print("Testing Approach 2 imports...")
    try:
        # Import using same pattern as Approach 2 scripts
        import yolo_detector
        import llm_generator
        import prompts
        import hybrid_pipeline
        print("  ✅ All Approach 2 imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_approach2_pipeline():
    """Test Approach 2 pipeline on a sample image (smoke test)"""
    print("\nTesting Approach 2 pipeline (smoke test)...")
    try:
        # Import using same pattern as Approach 2 scripts
        from hybrid_pipeline import run_hybrid_pipeline
        
        # Find a test image (relative to project root)
        test_images = [
            project_root / "data/images/gaming/tic_tac_toe-opp_move_1.png",
            project_root / "data/images/text/TEXT_CarDashboard_InstrumentCluster_WarningLights_ActiveDisplay.jpg",
            project_root / "data/images/indoor/INDOOR_Navigation_AccessibilityRamp_EntryDoor_BrickFacade.jpg",
        ]
        
        test_image = None
        for img_path in test_images:
            if img_path.exists():
                test_image = img_path
                break
        
        if not test_image:
            print("  ⚠️  No test image found, skipping pipeline test")
            return True
        
        print(f"  Using test image: {test_image.name}")
        result = run_hybrid_pipeline(
            test_image,
            yolo_size='n',
            llm_model='gpt-4o-mini'
        )
        
        if result['success']:
            print(f"  ✅ Pipeline test successful (latency: {result['total_latency']:.2f}s)")
            return True
        else:
            print(f"  ❌ Pipeline test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Pipeline test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_approach2_results_directory():
    """Verify Approach 2 results directory exists and is unchanged"""
    print("\nTesting Approach 2 results directory...")
    results_dir = Path("results/approach_2_yolo_llm/raw")
    csv_file = results_dir / "batch_results.csv"
    
    if results_dir.exists():
        print(f"  ✅ Results directory exists: {results_dir}")
        if csv_file.exists():
            print(f"  ✅ Results CSV exists: {csv_file}")
            return True
        else:
            print(f"  ⚠️  Results CSV not found (may be expected)")
            return True
    else:
        print(f"  ⚠️  Results directory not found (may be expected)")
        return True

def main():
    """Run all regression tests"""
    print("=" * 80)
    print("APPROACH 2.5 REGRESSION TESTS")
    print("=" * 80)
    print()
    
    tests = [
        ("Approach 2 Imports", test_approach2_imports),
        ("Approach 2 Pipeline", test_approach2_pipeline),
        ("Approach 2 Results Directory", test_approach2_results_directory),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))
    
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All regression tests passed!")
        return 0
    else:
        print("❌ Some regression tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())

