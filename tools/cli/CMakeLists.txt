set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# MSVC static lib linker needs all llama objects for transitive symbol resolution
if(MSVC)
    target_link_options(${TARGET} PRIVATE /WHOLEARCHIVE:llama)
endif()

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
