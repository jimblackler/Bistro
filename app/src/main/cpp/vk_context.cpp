#include "vk_context.h"

#include <android/log.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include "vulkan_debug.h"

#define LOG_TAG "BistroVk"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
constexpr VkDeviceSize kStagingSize = 64ull * 1024 * 1024;

bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    return std::any_of(exts.begin(), exts.end(),
                       [&](const VkExtensionProperties& e) { return strcmp(e.extensionName, name) == 0; });
}
}  // namespace

bool VulkanContext::init() {
    // ---- Instance ----
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> instExts(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, instExts.data());
    std::vector<const char*> extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    if (hasExtension(instExts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "Bistro";
    app.applicationVersion = 1;
    app.pEngineName = "BistroFlyThrough";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();
    VkResult res = vkCreateInstance(&ici, nullptr, &instance);
    if (res != VK_SUCCESS) {
        LOGE("vkCreateInstance failed (%d)", res);
        return false;
    }

    // ---- Physical device ----
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        LOGE("No Vulkan devices");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    physicalDevice = devices[0];
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    LOGI("GPU: %s, API %u.%u.%u", properties.deviceName, VK_API_VERSION_MAJOR(properties.apiVersion),
         VK_API_VERSION_MINOR(properties.apiVersion), VK_API_VERSION_PATCH(properties.apiVersion));
    if (properties.apiVersion < VK_API_VERSION_1_3) {
        LOGE("Vulkan 1.3 is required");
        return false;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    bool foundQueue = false;
    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            queueFamily = i;
            foundQueue = true;
            break;
        }
    }
    if (!foundQueue) {
        LOGE("No graphics+compute queue");
        return false;
    }

    // ---- Features ----
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f13.pNext = &f12;
    f12.pNext = &f11;
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &f13;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &supported);

    auto require = [](VkBool32 v, const char* what) {
        if (!v) LOGE("Missing required feature: %s", what);
        return v == VK_TRUE;
    };
    const std::pair<VkBool32, const char*> requirements[] = {
        {f13.dynamicRendering, "dynamicRendering"},
        {f13.synchronization2, "synchronization2"},
        {f12.drawIndirectCount, "drawIndirectCount"},
        {f12.runtimeDescriptorArray, "runtimeDescriptorArray"},
        {f12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound"},
        {f12.descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount"},
        {f12.shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing"},
        {f11.shaderDrawParameters, "shaderDrawParameters"},
        {supported.features.multiDrawIndirect, "multiDrawIndirect"},
        {supported.features.drawIndirectFirstInstance, "drawIndirectFirstInstance"},
        {supported.features.textureCompressionASTC_LDR, "textureCompressionASTC_LDR"},
    };
    bool ok = true;
    for (const auto& [value, what] : requirements) ok = require(value, what) && ok;
    if (!ok) return false;

    VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    e13.dynamicRendering = VK_TRUE;
    e13.synchronization2 = VK_TRUE;
    e13.maintenance4 = f13.maintenance4;
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e12.drawIndirectCount = VK_TRUE;
    e12.runtimeDescriptorArray = VK_TRUE;
    e12.descriptorBindingPartiallyBound = VK_TRUE;
    e12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    e12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    e12.descriptorIndexing = f12.descriptorIndexing;
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    e11.shaderDrawParameters = VK_TRUE;
    e13.pNext = &e12;
    e12.pNext = &e11;
    VkPhysicalDeviceFeatures2 enabled{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enabled.pNext = &e13;
    enabled.features.multiDrawIndirect = VK_TRUE;
    enabled.features.drawIndirectFirstInstance = VK_TRUE;
    enabled.features.textureCompressionASTC_LDR = VK_TRUE;
    enabled.features.samplerAnisotropy = supported.features.samplerAnisotropy;
    enabled.features.sampleRateShading = VK_FALSE;
    enabled.features.depthClamp = supported.features.depthClamp;
    maxAnisotropy = supported.features.samplerAnisotropy ? std::min(8.0f, properties.limits.maxSamplerAnisotropy) : 1.0f;

    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> devExts(count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, devExts.data());
    std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (hasExtension(devExts, VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enabled;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    dci.ppEnabledExtensionNames = deviceExtensions.data();
    res = vkCreateDevice(physicalDevice, &dci, nullptr, &device);
    if (res != VK_SUCCESS) {
        LOGE("vkCreateDevice failed (%d)", res);
        return false;
    }
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VulkanDebug::init(instance, device);
    VulkanDebug::setObjectName(device, instance, VK_OBJECT_TYPE_INSTANCE, "Bistro Vulkan 1.3 Instance");
    VulkanDebug::setObjectName(device, physicalDevice, VK_OBJECT_TYPE_PHYSICAL_DEVICE, properties.deviceName);
    VulkanDebug::setObjectName(device, device, VK_OBJECT_TYPE_DEVICE, properties.deviceName);
    VulkanDebug::setObjectName(device, queue, VK_OBJECT_TYPE_QUEUE, "Graphics+Compute Queue");
    return createUploadResources();
}

bool VulkanContext::createUploadResources() {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(device, &pci, nullptr, &oneShotPool_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(device, oneShotPool_, VK_OBJECT_TYPE_COMMAND_POOL, "One-Shot Upload Command Pool");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &fci, nullptr, &oneShotFence_) != VK_SUCCESS) return false;
    VulkanDebug::setObjectName(device, oneShotFence_, VK_OBJECT_TYPE_FENCE, "One-Shot Upload Fence");
    return createBuffer(kStagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_,
                        "Upload Staging Buffer");
}

void VulkanContext::destroy() {
    if (device) {
        vkDeviceWaitIdle(device);
        destroyBuffer(staging_);
        if (oneShotFence_) vkDestroyFence(device, oneShotFence_, nullptr);
        if (oneShotPool_) vkDestroyCommandPool(device, oneShotPool_, nullptr);
        oneShotFence_ = VK_NULL_HANDLE;
        oneShotPool_ = VK_NULL_HANDLE;
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, bool* found) const {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & props) == props) {
            if (found) *found = true;
            return i;
        }
    }
    if (found) *found = false;
    return 0;
}

bool VulkanContext::isFormatSupported(VkFormat format, VkFormatFeatureFlags features) const {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &p);
    return (p.optimalTilingFeatures & features) == features;
}

bool VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                                 GpuBuffer& out, const char* name) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
        LOGE("vkCreateBuffer failed for %s", name);
        return false;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        LOGE("vkAllocateMemory failed for %s (%llu bytes)", name, static_cast<unsigned long long>(req.size));
        return false;
    }
    allocatedBytes += req.size;
    vkBindBufferMemory(device, out.buffer, out.memory, 0);
    out.size = size;
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped);
    if (name) {
        VulkanDebug::setObjectName(device, out.buffer, VK_OBJECT_TYPE_BUFFER, name);
        std::string memName = std::string(name) + " Memory";
        VulkanDebug::setObjectName(device, out.memory, VK_OBJECT_TYPE_DEVICE_MEMORY, memName.c_str());
    }
    return true;
}

void VulkanContext::destroyBuffer(GpuBuffer& b) {
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = GpuBuffer{};
}

VkImageView VulkanContext::createView(VkImage image, VkFormat format, VkImageAspectFlags aspect, uint32_t baseMip,
                                      uint32_t mipCount, const char* name) {
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, baseMip, mipCount, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(device, &vci, nullptr, &view);
    if (name && view != VK_NULL_HANDLE) {
        VulkanDebug::setObjectName(device, view, VK_OBJECT_TYPE_IMAGE_VIEW, name);
    }
    return view;
}

bool VulkanContext::createImage(const ImageDesc& desc, GpuImage& out) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = desc.format;
    ici.extent = {desc.width, desc.height, 1};
    ici.mipLevels = desc.mipLevels;
    ici.arrayLayers = 1;
    ici.samples = desc.samples;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = desc.usage | (desc.transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0);
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &ici, nullptr, &out.image) != VK_SUCCESS) {
        LOGE("vkCreateImage failed for %s", desc.name ? desc.name : "?");
        return false;
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, out.image, &req);
    bool found = false;
    uint32_t type = 0;
    if (desc.transient) {
        type = findMemoryType(req.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, &found);
    }
    if (!found) type = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &mai, nullptr, &out.memory) != VK_SUCCESS) {
        LOGE("vkAllocateMemory failed for image %s", desc.name ? desc.name : "?");
        return false;
    }
    if (!found) allocatedBytes += req.size;
    vkBindImageMemory(device, out.image, out.memory, 0);
    std::string viewName = desc.name ? (std::string(desc.name) + " View") : std::string();
    out.view = createView(out.image, desc.format, desc.aspect, 0, desc.mipLevels,
                          desc.name ? viewName.c_str() : nullptr);
    out.format = desc.format;
    out.width = desc.width;
    out.height = desc.height;
    out.mipLevels = desc.mipLevels;
    if (desc.name) {
        VulkanDebug::setObjectName(device, out.image, VK_OBJECT_TYPE_IMAGE, desc.name);
        std::string memName = std::string(desc.name) + " Memory";
        VulkanDebug::setObjectName(device, out.memory, VK_OBJECT_TYPE_DEVICE_MEMORY, memName.c_str());
    }
    return true;
}

void VulkanContext::destroyImage(GpuImage& img) {
    if (img.view) vkDestroyImageView(device, img.view, nullptr);
    if (img.image) vkDestroyImage(device, img.image, nullptr);
    if (img.memory) vkFreeMemory(device, img.memory, nullptr);
    img = GpuImage{};
}

VkShaderModule VulkanContext::createShaderModule(const uint32_t* code, size_t size, const char* name) const {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = size;
    ci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &m) != VK_SUCCESS) {
        LOGE("vkCreateShaderModule failed for %s", name ? name : "?");
    } else if (name) {
        VulkanDebug::setObjectName(device, m, VK_OBJECT_TYPE_SHADER_MODULE, name);
    }
    return m;
}

VkCommandBuffer VulkanContext::beginOneShot(const char* name) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = oneShotPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);
    if (name) {
        VulkanDebug::setObjectName(device, cmd, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void VulkanContext::endOneShot(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(device, 1, &oneShotFence_);
    vkQueueSubmit(queue, 1, &si, oneShotFence_);
    vkWaitForFences(device, 1, &oneShotFence_, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(device, oneShotPool_, 1, &cmd);
}

void VulkanContext::recordPendingCopies(VkCommandBuffer cmd) {
    for (auto& c : pendingBufferCopies_) vkCmdCopyBuffer(cmd, staging_.buffer, c.dst, 1, &c.region);
    for (auto& c : pendingImageCopies_)
        vkCmdCopyBufferToImage(cmd, staging_.buffer, c.dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c.region);
    pendingBufferCopies_.clear();
    pendingImageCopies_.clear();
}

void VulkanContext::flushUploads() {
    if (pendingBufferCopies_.empty() && pendingImageCopies_.empty()) {
        stagingHead_ = 0;
        return;
    }
    VkCommandBuffer cmd = beginOneShot("Staged Upload Command Buffer");
    {
        ScopedCmdLabel label(cmd, "Flush Staged Uploads", DebugColors::Transfer);
        recordPendingCopies(cmd);
    }
    endOneShot(cmd);
    stagingHead_ = 0;
}

void VulkanContext::uploadBuffer(const GpuBuffer& dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size) {
    const auto* src = static_cast<const uint8_t*>(data);
    while (size > 0) {
        if (stagingHead_ >= kStagingSize) flushUploads();
        VkDeviceSize chunk = std::min(size, kStagingSize - stagingHead_);
        memcpy(static_cast<uint8_t*>(staging_.mapped) + stagingHead_, src, chunk);
        pendingBufferCopies_.push_back({dst.buffer, {stagingHead_, dstOffset, chunk}});
        stagingHead_ = (stagingHead_ + chunk + 15) & ~VkDeviceSize(15);
        src += chunk;
        dstOffset += chunk;
        size -= chunk;
    }
}

void VulkanContext::uploadImageLevel(const GpuImage& dst, uint32_t mip, uint32_t width, uint32_t height,
                                     const void* data, VkDeviceSize size) {
    if (size > kStagingSize) {
        LOGE("Image level too large for staging buffer");
        return;
    }
    if (stagingHead_ + size > kStagingSize) flushUploads();
    memcpy(static_cast<uint8_t*>(staging_.mapped) + stagingHead_, data, size);
    VkBufferImageCopy region{};
    region.bufferOffset = stagingHead_;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
    region.imageExtent = {width, height, 1};
    pendingImageCopies_.push_back({dst.image, region});
    stagingHead_ = (stagingHead_ + size + 15) & ~VkDeviceSize(15);
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout oldLayout,
                  VkImageLayout newLayout, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                  VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, uint32_t baseMip, uint32_t mipCount) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, baseMip, mipCount, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                   VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}
