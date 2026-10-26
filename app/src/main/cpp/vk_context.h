#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#include <cstdint>
#include <string>
#include <vector>

// Thin helpers around a Vulkan 1.3 instance/device plus basic buffer, image and upload management.

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 1;
};

struct ImageDesc {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t mipLevels = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    bool transient = false;  // prefer lazily-allocated memory (tile memory on TBDR GPUs)
    const char* name = nullptr;
};

class VulkanContext {
public:
    bool init();
    void destroy();

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, GpuBuffer& out,
                      const char* name);
    void destroyBuffer(GpuBuffer& b);
    bool createImage(const ImageDesc& desc, GpuImage& out);
    void destroyImage(GpuImage& img);
    VkImageView createView(VkImage image, VkFormat format, VkImageAspectFlags aspect, uint32_t baseMip,
                           uint32_t mipCount, const char* name = nullptr);

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, bool* found = nullptr) const;
    bool isFormatSupported(VkFormat format, VkFormatFeatureFlags features) const;
    VkShaderModule createShaderModule(const uint32_t* code, size_t size, const char* name = nullptr) const;

    // Synchronous single-use command buffers.
    VkCommandBuffer beginOneShot(const char* name = "One-Shot Command Buffer");
    void endOneShot(VkCommandBuffer cmd);

    // Staged uploads. Data is copied into a persistent staging buffer and flushed in batches.
    void uploadBuffer(const GpuBuffer& dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size);
    // Uploads one mip level; the image must be in TRANSFER_DST_OPTIMAL layout.
    void uploadImageLevel(const GpuImage& dst, uint32_t mip, uint32_t width, uint32_t height, const void* data,
                          VkDeviceSize size);
    void flushUploads();

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    float maxAnisotropy = 1.0f;
    VkDeviceSize allocatedBytes = 0;

private:
    bool createUploadResources();
    void recordPendingCopies(VkCommandBuffer cmd);

    VkCommandPool oneShotPool_ = VK_NULL_HANDLE;
    VkFence oneShotFence_ = VK_NULL_HANDLE;

    GpuBuffer staging_;
    VkDeviceSize stagingHead_ = 0;
    struct PendingBufferCopy {
        VkBuffer dst;
        VkBufferCopy region;
    };
    struct PendingImageCopy {
        VkImage dst;
        VkBufferImageCopy region;
    };
    std::vector<PendingBufferCopy> pendingBufferCopies_;
    std::vector<PendingImageCopy> pendingImageCopies_;
};

// Records an image layout transition using synchronization2.
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, uint32_t baseMip = 0,
                  uint32_t mipCount = VK_REMAINING_MIP_LEVELS);

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);
