# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.9)

project(manual-control-test)

set(CMAKE_CXX_STANDARD 17)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 
pkg_check_modules(Mosquitto IMPORTED_TARGET libmosquitto REQUIRED)
pkg_check_modules(Mosquittopp IMPORTED_TARGET libmosquittopp REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 

include(FindPackageHandleStandardArgs)


include_directories(/usr/include/driveless)

# control lib
find_library(CONTROL_LIB REQUIRED
    NAMES vehiclehal
    HINTS /usr/lib
)

# framework lib
find_library(FRAMEWORK_LIB REQUIRED
    NAMES drivelessfw
    HINTS /usr/lib
)

SET(TARGET_H
    ../../control/master_control_api.h
    ../../utils/file_utils.h
)

add_executable(
    remote_control
    manual_control_api_test.cpp
)

# add_executable(
#     vehicle_control_tst
#     vehicle_control_test.cpp
#     ../../control/vehicle_controller.cpp
# )

# link my-recognition to jetson-inference library
target_link_libraries(remote_control 
    PkgConfig::deps ${CONTROL_LIB} 
    PkgConfig::Mosquitto  
    PkgConfig::Mosquittopp 
    ${CMAKE_THREAD_LIBS_INIT}
    ${CONTROL_LIB}
    ${FRAMEWORK_LIB}

)
# target_link_libraries(vehicle_control_tst PkgConfig::deps ${CONTROL_LIB}  ${WIRINGPI_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})