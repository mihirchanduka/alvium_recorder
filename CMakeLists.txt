cmake_minimum_required(VERSION 3.16)

project(DualAlviumRecorder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VimbaX_2026-1/api/examples/cmake/vmb_cmake_prefix_paths.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/VimbaX_2026-1/api/examples/cmake/vmb_cmake_prefix_paths.cmake")
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV development files were not found. Install the system OpenCV dev package "
        "(for example, Fedora: 'sudo dnf install opencv-devel') and rerun CMake."
    )
endif()
find_package(Vmb REQUIRED COMPONENTS CPP NAMES Vmb vmb)

add_executable(alvium_recorder
    src/alvium_recorder.cpp
)

target_compile_definitions(alvium_recorder PRIVATE
    CV_IGNORE_DEBUG_BUILD_GUARD=1
    $<$<CONFIG:Debug>:_DEBUG>
)

target_link_libraries(alvium_recorder PRIVATE
    Vmb::CPP
    ${OpenCV_LIBS}
)

if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(alvium_recorder PRIVATE Threads::Threads)
    if(NOT APPLE)
        set(VIMBA_RUNTIME_DIRS "")
        foreach(_candidate
            "${CMAKE_CURRENT_SOURCE_DIR}/VimbaX_2026-1/api/lib"
            "${CMAKE_CURRENT_SOURCE_DIR}/VimbaX_2026-1/bin"
        )
            if(EXISTS "${_candidate}")
                list(APPEND VIMBA_RUNTIME_DIRS "${_candidate}")
            endif()
        endforeach()

        set_target_properties(alvium_recorder PROPERTIES
            BUILD_RPATH "${VIMBA_RUNTIME_DIRS}"
            INSTALL_RPATH "$ORIGIN;${VIMBA_RUNTIME_DIRS}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()
