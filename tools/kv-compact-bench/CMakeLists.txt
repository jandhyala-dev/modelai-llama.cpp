set(TARGET llama-kv-compact-bench)
add_executable(${TARGET} kv-compact-bench.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# MSVC single-pass linker needs WHOLE_ARCHIVE to resolve transitive symbols from llama
if(MSVC)
    target_link_libraries(${TARGET} PRIVATE common $<LINK_LIBRARY:WHOLE_ARCHIVE,llama> ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
