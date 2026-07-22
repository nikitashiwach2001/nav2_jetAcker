cmake_minimum_required(VERSION 3.8)
project(rl_nav_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(pluginlib REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# local onnxruntime shipped inside this package; pick the build matching the CPU arch
# (Jetson = aarch64, desktop = x64). Only the versioned .so is present in each.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ONNX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-aarch64-1.24.4)
else()
  set(ONNX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.24.4)
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNX_DIR}/lib/libonnxruntime.so.1.24.4
  INTERFACE_INCLUDE_DIRECTORIES ${ONNX_DIR}/include)

add_executable(onnx_node src/onnx_node.cpp)
add_library(rl_controller SHARED src/rl_controller.cpp)

target_link_libraries(onnx_node onnxruntime)
ament_target_dependencies(onnx_node rclcpp)

# RPATH so the plugin (loaded by controller_server) finds libonnxruntime at runtime
set_target_properties(rl_controller PROPERTIES
  BUILD_RPATH ${ONNX_DIR}/lib
  INSTALL_RPATH ${ONNX_DIR}/lib)
set_target_properties(onnx_node PROPERTIES
  BUILD_RPATH ${ONNX_DIR}/lib
  INSTALL_RPATH ${ONNX_DIR}/lib)

target_link_libraries(rl_controller onnxruntime)

ament_target_dependencies(rl_controller
  rclcpp
  nav2_core
  geometry_msgs
  nav_msgs
  sensor_msgs
  pluginlib
)

install(TARGETS onnx_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS rl_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

pluginlib_export_plugin_description_file(nav2_core rl_controller_plugin.xml)

# --- v2 controller (parallel plugin, does not replace rl_controller above) ---
add_library(rl_controller_v2 SHARED src/rl_controller_v2.cpp)

set_target_properties(rl_controller_v2 PROPERTIES
  BUILD_RPATH ${ONNX_DIR}/lib
  INSTALL_RPATH ${ONNX_DIR}/lib)

target_link_libraries(rl_controller_v2 onnxruntime)

ament_target_dependencies(rl_controller_v2
  rclcpp
  nav2_core
  geometry_msgs
  nav_msgs
  sensor_msgs
  pluginlib
)

install(TARGETS rl_controller_v2
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# pluginlib_export_plugin_description_file accumulates per plugin_category across multiple
# calls (verified in pluginlib's own CMake macro source) -- this does NOT overwrite or disturb
# the rl_controller_plugin.xml export above.
pluginlib_export_plugin_description_file(nav2_core rl_controller_v2_plugin.xml)

# --- v3 controller (parallel plugin, does not replace rl_controller/rl_controller_v2) ---
add_library(rl_controller_v3 SHARED src/rl_controller_v3.cpp)

set_target_properties(rl_controller_v3 PROPERTIES
  BUILD_RPATH ${ONNX_DIR}/lib
  INSTALL_RPATH ${ONNX_DIR}/lib)

target_link_libraries(rl_controller_v3 onnxruntime)

ament_target_dependencies(rl_controller_v3
  rclcpp
  nav2_core
  geometry_msgs
  nav_msgs
  sensor_msgs
  pluginlib
)

install(TARGETS rl_controller_v3
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

pluginlib_export_plugin_description_file(nav2_core rl_controller_v3_plugin.xml)

ament_package()
