# Host-side model validation tests
# Builds test_model_array.c against the firmware model data

cmake_minimum_required(VERSION 3.10)
project(spectra_host_tests C)

# Model data source and headers are in the main firmware directory
set(FIRMWARE_MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../spectra/main")

add_executable(test_model_array
    test_model_array.c
    "${FIRMWARE_MAIN_DIR}/model_data.cc"
)

target_include_directories(test_model_array PRIVATE
    "${FIRMWARE_MAIN_DIR}"
)

# Enable all warnings
if(MSVC)
    target_compile_options(test_model_array PRIVATE /W4)
else()
    target_compile_options(test_model_array PRIVATE -Wall -Wextra -Wpedantic)
endif()
