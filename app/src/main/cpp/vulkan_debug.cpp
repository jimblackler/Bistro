#include "vulkan_debug.h"
#include <android/log.h>

#define LOG_TAG "VulkanDebug"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

bool VulkanDebug::hasDebugUtils_ = false;
bool VulkanDebug::hasDebugMarker_ = false;

PFN_vkCmdBeginDebugUtilsLabelEXT VulkanDebug::pfnCmdBeginDebugUtilsLabelEXT = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT VulkanDebug::pfnCmdEndDebugUtilsLabelEXT = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT VulkanDebug::pfnCmdInsertDebugUtilsLabelEXT = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT VulkanDebug::pfnSetDebugUtilsObjectNameEXT = nullptr;
PFN_vkQueueBeginDebugUtilsLabelEXT VulkanDebug::pfnQueueBeginDebugUtilsLabelEXT = nullptr;
PFN_vkQueueEndDebugUtilsLabelEXT VulkanDebug::pfnQueueEndDebugUtilsLabelEXT = nullptr;

PFN_vkCmdDebugMarkerBeginEXT VulkanDebug::pfnCmdDebugMarkerBeginEXT = nullptr;
PFN_vkCmdDebugMarkerEndEXT VulkanDebug::pfnCmdDebugMarkerEndEXT = nullptr;
PFN_vkCmdDebugMarkerInsertEXT VulkanDebug::pfnCmdDebugMarkerInsertEXT = nullptr;
PFN_vkDebugMarkerSetObjectNameEXT VulkanDebug::pfnDebugMarkerSetObjectNameEXT = nullptr;

void VulkanDebug::init(VkInstance instance, VkDevice device) {
    if (instance != VK_NULL_HANDLE) {
        pfnCmdBeginDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"));
        pfnCmdEndDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
        pfnCmdInsertDebugUtilsLabelEXT = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkCmdInsertDebugUtilsLabelEXT"));
        pfnSetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT"));
        pfnQueueBeginDebugUtilsLabelEXT = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkQueueBeginDebugUtilsLabelEXT"));
        pfnQueueEndDebugUtilsLabelEXT = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(
            vkGetInstanceProcAddr(instance, "vkQueueEndDebugUtilsLabelEXT"));

        if (pfnCmdBeginDebugUtilsLabelEXT && pfnSetDebugUtilsObjectNameEXT) {
            hasDebugUtils_ = true;
            LOGI("VK_EXT_debug_utils initialized successfully!");
        }
    }

    if (device != VK_NULL_HANDLE && !hasDebugUtils_) {
        pfnCmdDebugMarkerBeginEXT = reinterpret_cast<PFN_vkCmdDebugMarkerBeginEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT"));
        pfnCmdDebugMarkerEndEXT = reinterpret_cast<PFN_vkCmdDebugMarkerEndEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT"));
        pfnCmdDebugMarkerInsertEXT = reinterpret_cast<PFN_vkCmdDebugMarkerInsertEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT"));
        pfnDebugMarkerSetObjectNameEXT = reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT"));

        if (pfnCmdDebugMarkerBeginEXT && pfnDebugMarkerSetObjectNameEXT) {
            hasDebugMarker_ = true;
            LOGI("VK_EXT_debug_marker initialized successfully!");
        }
    }
}

void VulkanDebug::setObjectNameInternal(VkDevice device, uint64_t handle, VkObjectType type, const char* name) {
    if (!name || handle == 0) return;

    if (pfnSetDebugUtilsObjectNameEXT && device != VK_NULL_HANDLE) {
        VkDebugUtilsObjectNameInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        pfnSetDebugUtilsObjectNameEXT(device, &info);
    } else if (pfnDebugMarkerSetObjectNameEXT && device != VK_NULL_HANDLE) {
        VkDebugMarkerObjectNameInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
        info.objectType = static_cast<VkDebugReportObjectTypeEXT>(type);
        info.object = handle;
        info.pObjectName = name;
        pfnDebugMarkerSetObjectNameEXT(device, &info);
    }
}

void VulkanDebug::cmdBeginLabel(VkCommandBuffer cmd, const char* name, const float color[4]) {
    if (!name || cmd == VK_NULL_HANDLE) return;

    if (pfnCmdBeginDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name;
        if (color) {
            for (int i = 0; i < 4; ++i) label.color[i] = color[i];
        } else {
            label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
        }
        pfnCmdBeginDebugUtilsLabelEXT(cmd, &label);
    } else if (pfnCmdDebugMarkerBeginEXT) {
        VkDebugMarkerMarkerInfoEXT marker{};
        marker.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
        marker.pMarkerName = name;
        if (color) {
            for (int i = 0; i < 4; ++i) marker.color[i] = color[i];
        } else {
            marker.color[0] = marker.color[1] = marker.color[2] = marker.color[3] = 1.0f;
        }
        pfnCmdDebugMarkerBeginEXT(cmd, &marker);
    }
}

void VulkanDebug::cmdEndLabel(VkCommandBuffer cmd) {
    if (cmd == VK_NULL_HANDLE) return;

    if (pfnCmdEndDebugUtilsLabelEXT) {
        pfnCmdEndDebugUtilsLabelEXT(cmd);
    } else if (pfnCmdDebugMarkerEndEXT) {
        pfnCmdDebugMarkerEndEXT(cmd);
    }
}

void VulkanDebug::cmdInsertLabel(VkCommandBuffer cmd, const char* name, const float color[4]) {
    if (!name || cmd == VK_NULL_HANDLE) return;

    if (pfnCmdInsertDebugUtilsLabelEXT) {
        VkDebugUtilsLabelEXT label{};
        label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name;
        if (color) {
            for (int i = 0; i < 4; ++i) label.color[i] = color[i];
        } else {
            label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
        }
        pfnCmdInsertDebugUtilsLabelEXT(cmd, &label);
    } else if (pfnCmdDebugMarkerInsertEXT) {
        VkDebugMarkerMarkerInfoEXT marker{};
        marker.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
        marker.pMarkerName = name;
        if (color) {
            for (int i = 0; i < 4; ++i) marker.color[i] = color[i];
        } else {
            marker.color[0] = marker.color[1] = marker.color[2] = marker.color[3] = 1.0f;
        }
        pfnCmdDebugMarkerInsertEXT(cmd, &marker);
    }
}

void VulkanDebug::queueBeginLabel(VkQueue queue, const char* name, const float color[4]) {
    if (!name || queue == VK_NULL_HANDLE || !pfnQueueBeginDebugUtilsLabelEXT) return;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    if (color) {
        for (int i = 0; i < 4; ++i) label.color[i] = color[i];
    } else {
        label.color[0] = label.color[1] = label.color[2] = label.color[3] = 1.0f;
    }
    pfnQueueBeginDebugUtilsLabelEXT(queue, &label);
}

void VulkanDebug::queueEndLabel(VkQueue queue) {
    if (queue == VK_NULL_HANDLE || !pfnQueueEndDebugUtilsLabelEXT) return;
    pfnQueueEndDebugUtilsLabelEXT(queue);
}
