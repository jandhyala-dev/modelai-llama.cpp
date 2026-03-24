set(TARGET llama-kv-compact-bench)
add_executable(${TARGET} kv-compact-bench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# MSVC static lib linker needs all llama objects for transitive symbol resolution
if(MSVC)
    target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:llama)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
