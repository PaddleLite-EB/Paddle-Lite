if (NOT LITE_WITH_FPGA)
    return()
endif()

set(LITE_FPGA_KD_PATH "${PADDLE_SOURCE_DIR}/lite/backends/fpga/KD")
set(LITE_FPGA_KD_LLAPI_PATH "${PADDLE_SOURCE_DIR}/lite/backends/fpga/KD/llapi")
set(LITE_FPGA_KD_PE_PATH "${PADDLE_SOURCE_DIR}/lite/backends/fpga/KD/pes")
set(LITE_FPGA_PATH "${PADDLE_SOURCE_DIR}/lite/backends/fpga")

message("fpga_kd_path ${LITE_FPGA_KD_PATH}")
message("fpga_path ${LITE_FPGA_PATH}")
file(GLOB KD_CPP "${LITE_FPGA_KD_PATH}/*.cpp")
file(GLOB PE_CPP "${LITE_FPGA_KD_PE_PATH}/*.cpp")
file(GLOB LLAPI_CPP "${LITE_FPGA_KD_LLAPI_PATH}/*.cpp")
file(GLOB FPGA_CPP "${LITE_FPGA_PATH}/*.cc")
set(FPGA_ALL_CPP "")
FOREACH(FILE_PATH ${KD_CPP})
    STRING(REGEX REPLACE ".+/(.+\\..*)" "\\1" FILE_NAME ${FILE_PATH})
    list(APPEND FPGA_ALL_CPP KD/${FILE_NAME})
ENDFOREACH(FILE_PATH)
FOREACH(FILE_PATH ${PE_CPP})
    STRING(REGEX REPLACE ".+/(.+\\..*)" "\\1" FILE_NAME ${FILE_PATH})
    list(APPEND FPGA_ALL_CPP KD/pes/${FILE_NAME})
ENDFOREACH(FILE_PATH)
FOREACH(FILE_PATH ${LLAPI_CPP})
    STRING(REGEX REPLACE ".+/(.+\\..*)" "\\1" FILE_NAME ${FILE_PATH})
    list(APPEND FPGA_ALL_CPP KD/llapi/${FILE_NAME})
ENDFOREACH(FILE_PATH)
FOREACH(FILE_PATH ${FPGA_CPP})
    STRING(REGEX REPLACE ".+/(.+\\..*)" "\\1" FILE_NAME ${FILE_PATH})
    list( APPEND FPGA_ALL_CPP ${FILE_NAME})
ENDFOREACH(FILE_PATH)
message("fpga kd: ${FPGA_ALL_CPP}")
cc_library(kernel_fpga SRCS ${FPGA_ALL_CPP})
#cc_library(kernel_fpga SRCS ${KD_CPP} ${FPGA_CPP})
cc_library(lite_tensor_fpga SRCS lite_tensor.cc DEPS memory)
cc_library(fpga_target_wrapper SRCS target_wrapper.cc DEPS kernel_fpga)


find_package(Git)
# 生成版本描述字符串类似 TAG-X-gHASH
execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --dirty --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit 日期，YYYY-MM-DD
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd --date=short
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit Hash
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 写入头文件宏定义的版本字符串，日期和 Hash
file(WRITE ${LITE_FPGA_KD_PATH}/repo_version.h
    "#define REPO_VERSION \"${GIT_REPO_VERSION}\"\n#define REPO_DATE \"${GIT_REPO_DATE}\"\n#define REPO_HASH \"${GIT_REPO_HASH}\"\n"
)

string (REGEX REPLACE ".*\t| .*" "" GIT_HASH ${GIT_REPO_HASH})
string (REGEX REPLACE "[\n\t\r]" "" GIT_VERSION ${GIT_REPO_VERSION})
string (REGEX REPLACE "[\n\t\r]" "" GIT_DATE ${GIT_REPO_DATE})

add_definitions( -DGIT_VERSION=\"${GIT_VERSION}\")
add_definitions( -DGIT_HASH=\"${GIT_HASH}\")
add_definitions( -DGIT_DATE=\"${GIT_DATE}\")

message("${GIT_REPO_VERSION}")
message("${GIT_REPO_DATE}")
message("${GIT_REPO_HASH}")
message("${LITE_FPGA_KD_PATH}/repo_version.h")
