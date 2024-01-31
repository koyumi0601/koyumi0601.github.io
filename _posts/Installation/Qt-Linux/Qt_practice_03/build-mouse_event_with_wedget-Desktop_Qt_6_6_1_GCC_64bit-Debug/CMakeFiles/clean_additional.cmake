# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles/mouse_event_with_wedget_autogen.dir/AutogenUsed.txt"
  "CMakeFiles/mouse_event_with_wedget_autogen.dir/ParseCache.txt"
  "mouse_event_with_wedget_autogen"
  )
endif()
