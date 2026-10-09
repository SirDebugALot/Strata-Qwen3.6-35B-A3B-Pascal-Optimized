# engines/qwen35moe - the Qwen3.5/3.6 MoE (qwen35moe) engine: its own CMake project, so the upstream engine's build
# (the repository root's CMakeLists.txt) stays as it is.  setup_qwen36.py builds it with Ninja:
#   cmake -G Ninja -S engines/qwen35moe -B build-qwen35moe -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)   # the GPUs in this PC; setup passes the exact list
endif()
project(strata_qwen35moe VERSION 0.1.0 LANGUAGES CXX CUDA)

# Experimental long-prompt path: multiply GGUF-native expert weights directly with
# llama.cpp's Pascal DP4A MMQ kernels.  It is deliberately opt-in so the proven
# phase-switch executable remains byte-for-byte reproducible.
option(SQ_ENABLE_PREFILL_MMQ "Use packed GGUF expert MMQ during batched prefill" OFF)
set(SQ_LLAMA_DIR "" CACHE PATH "Pinned llama.cpp checkout containing ggml/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_compile_definitions(SQ_VERSION="${PROJECT_VERSION}")
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/O2>
                      $<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus> $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  # the expert kernels per instruction set (cpu_moe.cpp picks one at run time; everything else stays x86-64)
  set(AVX512_FLAGS /arch:AVX512)
  set(AVX2_FLAGS /arch:AVX2)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
  set(AVX512_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512dq -mfma -mf16c)
  set(AVX2_FLAGS -mavx2 -mfma -mf16c)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --expt-relaxed-constexpr -Xcudafe --diag_suppress=177")
if(MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8 -Xcompiler=/wd4819 -Xcompiler=/Zc:preprocessor")
endif()

set_source_files_properties(src/cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "${AVX512_FLAGS}")
set_source_files_properties(src/cpu_avx2.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
set_source_files_properties(src/cpu_lm_head.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
set_source_files_properties(src/native_iq/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
set_source_files_properties(src/cpu_k23_avx2.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
set_source_files_properties(src/cpu_iq3_nibble.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")

add_library(sq_cpu STATIC src/quant_import.cpp src/gguf.cpp src/cpu_moe.cpp src/cpu_avx512.cpp src/cpu_avx2.cpp src/cpu_lm_head.cpp src/native_iq/iq_avx2.cpp src/cpu_k23_avx2.cpp)
target_sources(sq_cpu PRIVATE src/cpu_iq3_nibble.cpp src/cpu_iq3_cache.cpp)
target_include_directories(sq_cpu PUBLIC src)
target_link_libraries(sq_cpu PUBLIC CUDA::cudart_static Threads::Threads)

add_library(sq_core STATIC
  src/model.cpp
  src/engine.cpp
  src/sampler.cpp
  src/kernels.cu
  src/dense_q6.cu
  src/mtp_subset.cu
  src/kv_int8.cu
  src/prefill.cu
)
target_include_directories(sq_core PUBLIC src)
target_link_libraries(sq_core PUBLIC sq_cpu CUDA::cudart_static CUDA::cublas)
set_target_properties(sq_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(SQ_ENABLE_PREFILL_MMQ)
  if(NOT SQ_LLAMA_DIR OR NOT EXISTS "${SQ_LLAMA_DIR}/ggml/src/ggml-cuda/mmq.cuh")
    message(FATAL_ERROR "SQ_ENABLE_PREFILL_MMQ requires -DSQ_LLAMA_DIR=<pinned llama.cpp checkout>")
  endif()
  enable_language(C)
  # Only ggml-base is linked; ggml-cuda itself stays disabled because this engine
  # supplies the small host shim and compiles just MMQ + activation quantization.
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_METAL OFF CACHE BOOL "" FORCE)
  set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${SQ_LLAMA_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

  get_filename_component(_sq_qwen_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  set(_sq_ggcuda "${SQ_LLAMA_DIR}/ggml/src/ggml-cuda")
  set(_sq_mmq_types q2_0 q2_k q3_k iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0)
  add_library(sq_prefill_mmq STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/prefill_mmq.cu"
    "${_sq_qwen_root}/src/prefill/ggml_cuda_host.cu"
    "${_sq_ggcuda}/quantize.cu"
  )
  foreach(_t IN LISTS _sq_mmq_types)
    target_sources(sq_prefill_mmq PRIVATE "${_sq_ggcuda}/template-instances/mmq-instance-${_t}.cu")
  endforeach()
  target_include_directories(sq_prefill_mmq PRIVATE
    "${SQ_LLAMA_DIR}/ggml/include" "${SQ_LLAMA_DIR}/ggml/src" "${_sq_ggcuda}"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_options(sq_prefill_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
  if(MSVC)
    target_compile_options(sq_prefill_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
  endif()
  target_link_libraries(sq_prefill_mmq PUBLIC ggml-base CUDA::cudart_static)
  target_compile_definitions(sq_prefill_mmq PRIVATE STRATA_MMQ_Q23_K=1)
  target_link_libraries(sq_core PUBLIC sq_prefill_mmq)
  target_compile_definitions(sq_core PUBLIC SQ_PREFILL_MMQ=1)
endif()

add_executable(strata-qwen35moe src/main.cpp)
target_link_libraries(strata-qwen35moe PRIVATE sq_core)

# Explicit experimental target; building it does not relink the normal executable.
add_executable(strata-qwen35moe-aggressive EXCLUDE_FROM_ALL src/main.cpp)
target_link_libraries(strata-qwen35moe-aggressive PRIVATE sq_core)

add_executable(strata-qwen35moe-cachelab EXCLUDE_FROM_ALL src/main.cpp)
target_link_libraries(strata-qwen35moe-cachelab PRIVATE sq_core)

add_executable(strata-qwen35moe-optimized EXCLUDE_FROM_ALL src/main.cpp)
target_link_libraries(strata-qwen35moe-optimized PRIVATE sq_core)
target_compile_definitions(strata-qwen35moe-optimized PRIVATE SQ_ENGINE_TAG="qwen35moe-native-pascal-20261008")

add_executable(test_iq_native EXCLUDE_FROM_ALL src/iq_native.cu)
target_link_libraries(test_iq_native PRIVATE sq_cpu CUDA::cudart_static)

add_executable(test_iq_multitoken EXCLUDE_FROM_ALL src/test_iq_multitoken.cpp)
target_link_libraries(test_iq_multitoken PRIVATE sq_cpu)

add_executable(test_q8k_pack EXCLUDE_FROM_ALL src/test_q8k_pack.cpp)
target_link_libraries(test_q8k_pack PRIVATE sq_cpu)

add_executable(test_gpu_expert_reuse EXCLUDE_FROM_ALL src/test_gpu_expert_reuse.cu)
target_link_libraries(test_gpu_expert_reuse PRIVATE sq_core)

add_executable(test_native_layout_perf EXCLUDE_FROM_ALL src/test_gpu_expert_reuse.cu)
target_compile_definitions(test_native_layout_perf PRIVATE SQ_TEST_NATIVE_LAYOUT=1)
target_link_libraries(test_native_layout_perf PRIVATE sq_core)

add_executable(test_prompt_batch EXCLUDE_FROM_ALL src/test_prompt_batch.cpp)
target_link_libraries(test_prompt_batch PRIVATE sq_core)

add_executable(test_kv_int8 EXCLUDE_FROM_ALL src/test_kv_int8.cu)
target_link_libraries(test_kv_int8 PRIVATE sq_core)

add_executable(test_lazy_kv EXCLUDE_FROM_ALL src/test_lazy_kv.cpp)
target_link_libraries(test_lazy_kv PRIVATE sq_core)

add_executable(test_lazy_kv_pressure EXCLUDE_FROM_ALL src/test_lazy_kv_pressure.cpp)
target_link_libraries(test_lazy_kv_pressure PRIVATE sq_core)

add_executable(test_kv_quality EXCLUDE_FROM_ALL src/test_kv_quality.cpp)
target_link_libraries(test_kv_quality PRIVATE sq_core)

add_executable(test_native_dense_quality EXCLUDE_FROM_ALL src/test_native_dense_quality.cpp)
target_link_libraries(test_native_dense_quality PRIVATE sq_core)

add_executable(test_spec_stop EXCLUDE_FROM_ALL src/test_spec_stop.cpp)
target_link_libraries(test_spec_stop PRIVATE sq_core)

add_executable(test_mtp_subset EXCLUDE_FROM_ALL src/test_mtp_subset.cu)
target_link_libraries(test_mtp_subset PRIVATE sq_core)

add_executable(test_actq_readonly EXCLUDE_FROM_ALL src/test_actq_readonly.cu)
target_link_libraries(test_actq_readonly PRIVATE sq_core)

add_executable(test_event_moe_graph EXCLUDE_FROM_ALL src/test_event_moe_graph.cpp)
target_link_libraries(test_event_moe_graph PRIVATE sq_core)

add_executable(test_dense_q6 EXCLUDE_FROM_ALL src/test_dense_q6.cu)
target_link_libraries(test_dense_q6 PRIVATE sq_core)

add_executable(test_cpu_kernels src/test_cpu_kernels.cpp)
target_link_libraries(test_cpu_kernels PRIVATE sq_cpu)

add_executable(test_cpu_lm_head src/test_cpu_lm_head.cpp)
target_link_libraries(test_cpu_lm_head PRIVATE sq_cpu)

add_executable(test_argmax_no_prob EXCLUDE_FROM_ALL src/test_argmax_no_prob.cpp)
target_link_libraries(test_argmax_no_prob PRIVATE sq_core)

add_executable(test_router_mailbox EXCLUDE_FROM_ALL src/test_router_mailbox.cpp)
target_link_libraries(test_router_mailbox PRIVATE sq_core)

add_executable(test_event_graph_cache EXCLUDE_FROM_ALL src/test_event_graph_cache.cpp)
target_link_libraries(test_event_graph_cache PRIVATE sq_core)

add_executable(test_mtp_last_row EXCLUDE_FROM_ALL src/test_mtp_last_row.cpp)
target_link_libraries(test_mtp_last_row PRIVATE sq_core)

add_executable(test_cpu_k23_multitoken EXCLUDE_FROM_ALL src/test_cpu_k23_multitoken.cpp)
target_link_libraries(test_cpu_k23_multitoken PRIVATE sq_cpu)

add_executable(test_iq_inline_rows EXCLUDE_FROM_ALL src/test_iq_inline_rows.cpp)
target_link_libraries(test_iq_inline_rows PRIVATE sq_cpu)

add_executable(test_cpu_completion_wait EXCLUDE_FROM_ALL src/test_cpu_completion_wait.cpp)
target_link_libraries(test_cpu_completion_wait PRIVATE sq_cpu)

add_executable(test_miss_stream EXCLUDE_FROM_ALL src/test_miss_stream.cpp)
target_link_libraries(test_miss_stream PRIVATE sq_core)

add_executable(test_streaming_quality EXCLUDE_FROM_ALL src/test_streaming_quality.cpp)
target_link_libraries(test_streaming_quality PRIVATE sq_core)

# Standalone CPU research probe; never linked into the inference engine.
set_source_files_properties(src/iq3_predecode_probe.cpp PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
add_executable(test_iq3_predecode EXCLUDE_FROM_ALL src/test_iq3_predecode.cpp src/iq3_predecode_probe.cpp)
target_link_libraries(test_iq3_predecode PRIVATE sq_cpu)

add_executable(test_cpu_iq3_cache EXCLUDE_FROM_ALL src/test_cpu_iq3_cache.cpp)
target_link_libraries(test_cpu_iq3_cache PRIVATE sq_cpu)

add_executable(test_cpu_iq3_wide EXCLUDE_FROM_ALL src/test_cpu_iq3_wide.cpp)
target_link_libraries(test_cpu_iq3_wide PRIVATE sq_cpu)

add_executable(test_gdn_prepare EXCLUDE_FROM_ALL src/test_gdn_prepare.cpp)
target_link_libraries(test_gdn_prepare PRIVATE sq_core)

add_executable(test_cpu_phase_profile EXCLUDE_FROM_ALL src/test_cpu_phase_profile.cpp)
target_link_libraries(test_cpu_phase_profile PRIVATE sq_cpu)

add_executable(test_prompt_full_graph EXCLUDE_FROM_ALL src/test_prompt_full_graph.cpp)
target_link_libraries(test_prompt_full_graph PRIVATE sq_core)
