set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# MSVC single-pass linker needs WHOLE_ARCHIVE to resolve transitive symbols from llama
if(MSVC)
    target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common $<LINK_LIBRARY:WHOLE_ARCHIVE,llama> ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(${TARGET} PRIVATE server-context PUBLIC common ${CMAKE_THREAD_LIBS_INIT})
endif()

include_directories(../server)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
