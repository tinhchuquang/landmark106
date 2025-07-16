cmake_minimum_required(VERSION 3.10)
project(trt)

set(CMAKE_CXX_STANDARD 17)

# Tìm OpenCV
find_package(OpenCV REQUIRED)

# Bao gồm thư mục scrfd và thư mục hiện tại
include_directories(/usr/local/cuda/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(
    ./scrfd
    ./utils
    ./pfld
    ./
    .
)

add_executable(pfld scrfd/scrfd.h scrfd/scrfd_postprocess.h utils/utils.cpp)

# Link OpenCV + TensorRT + CUDA runtime (thường hệ thống đã có)
target_link_directories(scrfd PUBLIC /usr/local/cuda/lib64)
target_link_libraries(scrfd
    ${OpenCV_LIBS}
    nvinfer
    cudart
)

# Nếu build trên Jetson hoặc TensorRT custom, có thể cần nvinfer_plugin, thêm nếu báo thiếu symbol:
# target_link_libraries(scrfd nvinfer_plugin)
