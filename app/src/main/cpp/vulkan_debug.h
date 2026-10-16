#pragma once

#include <vulkan/vulkan.h>
#include <type_traits>
#include <cstdint>

class VulkanDebug {
public:
    static void init(VkInstance instance, VkDevice device);

    template <typename T>
    static void setObjectName(VkDevice device, T handle, VkObjectType type, const char* name) {
        uint64_t handle64 = toUint64(handle);
        setObjectNameInternal(device, handle64, type, name);
    }

    static void cmdBeginLabel(VkCommandBuffer cmd, const char* name, const float color[4] = nullptr);
    static void cmdEndLabel(VkCommandBuffer cmd);
    static void cmdInsertLabel(VkCommandBuffer cmd, const char* name, const float color[4] = nullptr);

    static void queueBeginLabel(VkQueue queue, const char* name, const float color[4] = nullptr);
    static void queueEndLabel(VkQueue queue);

    [[nodiscard]] static bool isDebugUtilsSupported() { return hasDebugUtils_; }
    [[nodiscard]] static bool isDebugMarkerSupported() { return hasDebugMarker_; }

private:
    template <typename T>
    static inline uint64_t toUint64(T handle) {
        if constexpr (std::is_pointer_v<T>) {
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
        } else {
            return static_cast<uint64_t>(handle);
        }
    }

    static void setObjectNameInternal(VkDevice device, uint64_t handle, VkObjectType type, const char* name);

    static bool hasDebugUtils_;
    static bool hasDebugMarker_;

    static PFN_vkCmdBeginDebugUtilsLabelEXT pfnCmdBeginDebugUtilsLabelEXT;
    static PFN_vkCmdEndDebugUtilsLabelEXT pfnCmdEndDebugUtilsLabelEXT;
    static PFN_vkCmdInsertDebugUtilsLabelEXT pfnCmdInsertDebugUtilsLabelEXT;
    static PFN_vkSetDebugUtilsObjectNameEXT pfnSetDebugUtilsObjectNameEXT;
    static PFN_vkQueueBeginDebugUtilsLabelEXT pfnQueueBeginDebugUtilsLabelEXT;
    static PFN_vkQueueEndDebugUtilsLabelEXT pfnQueueEndDebugUtilsLabelEXT;

    static PFN_vkCmdDebugMarkerBeginEXT pfnCmdDebugMarkerBeginEXT;
    static PFN_vkCmdDebugMarkerEndEXT pfnCmdDebugMarkerEndEXT;
    static PFN_vkCmdDebugMarkerInsertEXT pfnCmdDebugMarkerInsertEXT;
    static PFN_vkDebugMarkerSetObjectNameEXT pfnDebugMarkerSetObjectNameEXT;
};

// RAII helper for scoped command buffer labels
struct ScopedCmdLabel {
    VkCommandBuffer cmd;
    bool active = false;

    ScopedCmdLabel(VkCommandBuffer commandBuffer, const char* name, const float color[4] = nullptr)
        : cmd(commandBuffer), active(commandBuffer != VK_NULL_HANDLE) {
        if (active) VulkanDebug::cmdBeginLabel(cmd, name, color);
    }

    ~ScopedCmdLabel() {
        if (active) VulkanDebug::cmdEndLabel(cmd);
    }

    ScopedCmdLabel(const ScopedCmdLabel&) = delete;
    ScopedCmdLabel& operator=(const ScopedCmdLabel&) = delete;
};

// Color presets for profiler and debugger visualization
namespace DebugColors {
    constexpr float Frame[4]       = {0.20f, 0.45f, 0.85f, 1.0f}; // Royal Blue
    constexpr float RenderPass[4]  = {0.15f, 0.25f, 0.65f, 1.0f}; // Deep Indigo
    constexpr float Draw[4]        = {0.95f, 0.55f, 0.15f, 1.0f}; // Amber / Orange
    constexpr float Transition[4]  = {0.70f, 0.30f, 0.85f, 1.0f}; // Violet
    constexpr float Transfer[4]    = {0.20f, 0.75f, 0.35f, 1.0f}; // Emerald Green
    constexpr float Marker[4]      = {0.80f, 0.80f, 0.85f, 1.0f}; // Silver
    constexpr float Queue[4]       = {0.10f, 0.65f, 0.90f, 1.0f}; // Sky Blue
}
