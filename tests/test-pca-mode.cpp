// Test file for PCA calibration mode - tests requirements:
// 1. mode_from_string("pca") must be accepted and return generic PCA behavior
// 2. Legacy strings turbo3_pca, turbo4_pca, etc. must all map to generic PCA
// 3. mode_to_string(PCA) must return "pca"
// 4. --kv-pca-calibrate should set mode to "pca", not "turbo3_pca"

#include <cassert>
#include <cstdio>
#include <string>
#include "common.h"
#include "arg.h"
#include "llama-kv-empvar-calibration.h"

int main() {
    printf("=== PCA Calibration Mode Tests ===\n\n");
    
    // Test 1: mode_from_string("pca") should work and return PCA mode
    {
        printf("test_mode_from_string_pca: ...\n");
        auto mode = llama_kv_empvar_calibration::mode_from_string("pca");
        assert(mode == llama_kv_empvar_calibration::mode_t::PCA);
        printf("test_mode_from_string_pca: PASS\n");
    }
    
    // Test 2: mode_to_string(PCA) should return "pca"
    {
        printf("test_mode_to_string_pca: ...\n");
        const char* mode_str = llama_kv_empvar_calibration::mode_to_string(
            llama_kv_empvar_calibration::mode_t::PCA);
        assert(mode_str != nullptr);
        assert(std::string(mode_str) == "pca");
        printf("test_mode_to_string_pca: PASS (mode_str = \"%s\")\n", mode_str);
    }
    
    // Test 3: Legacy strings map to PCA
    {
        printf("test_legacy_strings: ...\n");
        auto mode3 = llama_kv_empvar_calibration::mode_from_string("turbo3_pca");
        auto mode4 = llama_kv_empvar_calibration::mode_from_string("turbo4_pca");
        auto mode4333 = llama_kv_empvar_calibration::mode_from_string("turbo4333_pca");
        auto mode4322 = llama_kv_empvar_calibration::mode_from_string("turbo4322_pca");
        auto mode4211 = llama_kv_empvar_calibration::mode_from_string("turbo4211_pca");
        
        assert(mode3 == llama_kv_empvar_calibration::mode_t::PCA);
        assert(mode4 == llama_kv_empvar_calibration::mode_t::PCA);
        assert(mode4333 == llama_kv_empvar_calibration::mode_t::PCA);
        assert(mode4322 == llama_kv_empvar_calibration::mode_t::PCA);
        assert(mode4211 == llama_kv_empvar_calibration::mode_t::PCA);
        printf("test_legacy_strings: PASS\n");
    }
    
    // Test 4: --kv-pca-calibrate sets mode to "pca"
    {
        printf("test_kv_pca_calibrate_flag: ...\n");
        common_params params;
        std::vector<std::string> argv_str = {"test", "-m", "dummy.gguf", "--kv-pca-calibrate"};
        std::vector<char*> argv;
        for (auto & s : argv_str) {
            argv.push_back(s.data());
        }
        bool ok = common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_PERPLEXITY);
        assert(ok);
        assert(params.kv_empvar_calibrate == true);
        assert(params.kv_calibration_mode == "pca");
        printf("test_kv_pca_calibrate_flag: PASS (kv_calibration_mode = \"%s\")\n", 
               params.kv_calibration_mode.c_str());
    }
    
    printf("\n=== All tests passed ===\n");
    return 0;
}