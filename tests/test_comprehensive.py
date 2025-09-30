"""
Comprehensive Test Suite

Master test runner that imports and runs all verification tests.
"""

import pytest
import sys
from pathlib import Path

# Add the src directory to the path so we can import w4a
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import all test modules
try:
    from test_spaces import TestObservationSpace, TestActionSpace, TestActionMasking as SpaceActionMasking
    from test_rewards import TestRewardCustomization, TestRewardSignals
    from test_lifecycle import TestEnvironmentProperties, TestTerminationConditions, TestEnvironmentIntegration
    from test_action_masking import TestActionMaskStructure, TestMaskValidation
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Warning: Could not import all test modules: {e}")
    IMPORTS_SUCCESSFUL = False


class TestComprehensive:
    """Comprehensive test suite runner"""
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Test module imports failed")
    def test_observation_space_comprehensive(self):
        """Run observation space tests"""
        print("Running observation space tests...")
        
        test_obs = TestObservationSpace()
        test_obs.test_observation_consistency()
        test_obs.test_observation_features_bounded()
        
        print("All observation space tests passed")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Test module imports failed")
    def test_action_space_comprehensive(self):
        """Run action space tests"""
        print("Running action space tests...")
        
        test_action = TestActionSpace()
        test_action.test_action_space_structure()
        test_action.test_action_space_bounds()
        test_action.test_action_sampling()
        
        print("All action space tests passed")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Test module imports failed")
    def test_reward_system_comprehensive(self):
        """Run reward system tests"""
        print("Running reward system tests...")
        
        test_custom = TestRewardCustomization()
        test_custom.test_wrapper_reward_modification()
        
        test_signals = TestRewardSignals()
        test_signals.test_reward_changes_with_actions()
        
        print("All reward system tests passed")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Test module imports failed")
    def test_lifecycle_comprehensive(self):
        """Run lifecycle tests"""
        print("Running lifecycle tests...")
        
        test_properties = TestEnvironmentProperties()
        test_properties.test_gymnasium_interface_compliance()
        
        test_termination = TestTerminationConditions()
        test_termination.test_time_based_truncation()
        test_termination.test_manual_termination_conditions()
        
        test_integration = TestEnvironmentIntegration()
        test_integration.test_random_agent_integration()
        
        print("All lifecycle tests passed")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason="Test module imports failed")
    def test_action_masking_comprehensive(self):
        """Run action masking tests"""
        print("Running action masking tests...")
        
        test_structure = TestActionMaskStructure()
        test_structure.test_action_type_masks()
        test_structure.test_controllable_entity_masks()
        test_structure.test_detected_target_masks()
        test_structure.test_entity_target_matrix_masks()
        
        test_validation = TestMaskValidation()
        test_validation.test_valid_action_respects_masks()
        
        # Also run the consistency test from spaces
        test_space_masks = SpaceActionMasking()
        test_space_masks.test_masks_consistency_during_episode()
        
        print("All action masking tests passed")


def run_comprehensive_tests():
    """Run all tests as a standalone function"""
    print("=" * 60)
    print("COMPREHENSIVE FUNCTIONALITY TEST SUITE")
    print("=" * 60)
    
    if not IMPORTS_SUCCESSFUL:
        print("Cannot run tests - import failures")
        return False
    
    try:
        # Create test instance
        test_suite = TestComprehensive()
        
        # Run all test categories
        test_suite.test_observation_space_comprehensive()
        test_suite.test_action_space_comprehensive()
        test_suite.test_reward_system_comprehensive()
        test_suite.test_lifecycle_comprehensive()
        test_suite.test_action_masking_comprehensive()
        
        print("=" * 60)
        print("ALL COMPREHENSIVE TESTS PASSED")
        print("=" * 60)
        print()
        print("Test Coverage Summary:")
        print("- Observation Space: Consistency, bounds, normalization")
        print("- Action Space: Structure, bounds, sampling")
        print("- Reward System: Customization, signal variation")
        print("- Episode Lifecycle: Gymnasium compliance, termination")
        print("- Action Masking: Structure, validation, consistency")
        print()
        print("Your environment is ready for training")
        
        return True
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)