cmake_minimum_required(VERSION 3.22.1)

project("bistro")

set(SHADER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")

find_program(GLSLC glslc HINTS
    ${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}
    ${CMAKE_ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}
    ${ANDROID_NDK}/shader-tools/darwin-x86_64
    ${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64
)

# Compiles each GLSL shader into a C array header (<name>.spv.h) that is embedded in the library.
set(SHADERS
    cull.comp
    hiz.comp
    exposure.comp
    scene.vert
    scene.frag
    shadow.vert
    shadow.frag
    fullscreen.vert
    sky.frag
    bloom_down.frag
    bloom_up.frag
    tonemap.frag
    taa.frag
    loading.frag
    ao.comp
    ao_blur.comp
)
set(SHADER_HEADERS "")
foreach(SHADER ${SHADERS})
    set(OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER}.spv.h")
    add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND ${GLSLC} --target-env=vulkan1.3 -O -mfmt=c -I ${SHADER_SRC_DIR}
                ${SHADER_SRC_DIR}/${SHADER} -o ${OUT}
        DEPENDS ${SHADER_SRC_DIR}/${SHADER} ${SHADER_SRC_DIR}/common.glsl
        COMMENT "Compiling ${SHADER} to SPIR-V header"
    )
    list(APPEND SHADER_HEADERS ${OUT})
endforeach()

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        render_thread.cpp
        renderer.cpp
        scene.cpp
        camera_path.cpp
        vk_context.cpp
        vulkan_debug.cpp
        ${SHADER_HEADERS})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/shaders)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        vulkan
        log)
