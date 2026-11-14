// Copyright 2025-current Getnamo.
//
// Vulkan entry points for ggml-vulkan on Android.
//
// ggml-vulkan resolves nearly every Vulkan function through a dispatcher it initializes from the global
// vkGetInstanceProcAddr, and calls only vkGetDeviceProcAddr, vkGetPhysicalDeviceFeatures2 and vkCmdCopyBuffer
// directly. Defining those four here, instead of binding them from libvulkan, does two things:
//
// 1. UE links Android's libvulkan stub for API 26, which only exports Vulkan 1.0 entry points, and Android's
//    dynamic linker binds every symbol when the game's .so loads. Linking a newer stub would keep the game from
//    starting on Android 8 devices; resolving at call time doesn't (ggml-vulkan needs a Vulkan 1.2 device anyway).
//
// 2. llama.cpp can use another Vulkan driver than the engine. Qualcomm's driver fails to build some ggml-vulkan
//    shaders on Adreno GPUs (llama.cpp issue 28635); Mesa's Turnip driver builds them. With the environment
//    variable LLAMA_VULKAN_DRIVER set to a driver library (a file name such as libvulkan_freedreno.so packaged
//    with the app, or a full path), llama.cpp loads it as an Android Vulkan HAL module and talks to it directly.
//    UE's renderer keeps using the system driver: VulkanRHI loads its own entry points with dlsym.
//
// Also compiles outside UE (-DLLAMA_VULKAN_SHIM_STANDALONE), e.g. into llama.cpp's tools for testing.

#ifdef LLAMA_VULKAN_SHIM_STANDALONE
#define PLATFORM_ANDROID 1
#else
#include "CoreMinimal.h"
#endif

#if PLATFORM_ANDROID

#include <android/log.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <vulkan/vulkan.h>

namespace
{
    // Android Vulkan HAL module ABI (hardware/libhardware: hardware.h, hwvulkan.h)
    struct FHwModule;

    struct FHwDevice
    {
        uint32_t Tag;
        uint32_t Version;
        FHwModule* Module;
        uintptr_t Reserved[12];
        int (*Close)(FHwDevice* Device);
    };

    struct FHwModuleMethods
    {
        int (*Open)(const FHwModule* Module, const char* Id, FHwDevice** Device);
    };

    struct FHwModule
    {
        uint32_t Tag;
        uint16_t ModuleApiVersion;
        uint16_t HalApiVersion;
        const char* Id;
        const char* Name;
        const char* Author;
        FHwModuleMethods* Methods;
        void* Dso;
        uintptr_t Reserved[32 - 7];
    };

    struct FHwVulkanDevice
    {
        FHwDevice Common;
        PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
        PFN_vkCreateInstance CreateInstance;
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    };

    struct FVulkanDriver
    {
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
        PFN_vkCreateInstance CreateInstance = nullptr;
        void* SystemLibrary = nullptr;
    };

    const char* LogTag = "LlamaVulkan";

    // Counts the GPUs a driver supports (Turnip: Qualcomm Adreno only), with a throwaway instance
    uint32_t CountPhysicalDevices(const FHwVulkanDevice* VulkanDevice)
    {
        VkApplicationInfo AppInfo = {};
        AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        AppInfo.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo CreateInfo = {};
        CreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        CreateInfo.pApplicationInfo = &AppInfo;

        uint32_t Count = 0;
        VkInstance Probe = VK_NULL_HANDLE;
        if (VulkanDevice->CreateInstance && VulkanDevice->CreateInstance(&CreateInfo, nullptr, &Probe) == VK_SUCCESS)
        {
            const PFN_vkEnumeratePhysicalDevices Enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
                VulkanDevice->GetInstanceProcAddr(Probe, "vkEnumeratePhysicalDevices"));
            const PFN_vkDestroyInstance Destroy = reinterpret_cast<PFN_vkDestroyInstance>(
                VulkanDevice->GetInstanceProcAddr(Probe, "vkDestroyInstance"));
            if (Enumerate)
            {
                Enumerate(Probe, &Count, nullptr);
            }
            if (Destroy)
            {
                Destroy(Probe, nullptr);
            }
        }
        return Count;
    }

    // Loads a Vulkan driver library as an Android Vulkan HAL module; false if it can't be used on this device
    bool TryHalDriver(const char* LibraryName, bool bRequested, FVulkanDriver& Driver)
    {
        void* Library = dlopen(LibraryName, RTLD_NOW | RTLD_LOCAL);
        if (!Library)
        {
            if (bRequested)
            {
                __android_log_print(ANDROID_LOG_WARN, LogTag, "Could not load the Vulkan driver %s (%s), using the system driver", LibraryName, dlerror());
            }
            return false;
        }

        const FHwModule* Module = static_cast<const FHwModule*>(dlsym(Library, "HMI"));
        FHwDevice* Device = nullptr;
        if (!Module || !Module->Methods || !Module->Methods->Open || Module->Methods->Open(Module, "vk0", &Device) != 0 || !Device)
        {
            __android_log_print(ANDROID_LOG_WARN, LogTag, "%s is not a Vulkan HAL module, using the system driver", LibraryName);
            return false;
        }

        const FHwVulkanDevice* VulkanDevice = reinterpret_cast<const FHwVulkanDevice*>(Device);
        if (!VulkanDevice->GetInstanceProcAddr || CountPhysicalDevices(VulkanDevice) == 0)
        {
            __android_log_print(ANDROID_LOG_INFO, LogTag, "The Vulkan driver %s doesn't support this GPU, using the system driver", LibraryName);
            return false;
        }

        Driver.GetInstanceProcAddr = VulkanDevice->GetInstanceProcAddr;
        Driver.CreateInstance = VulkanDevice->CreateInstance;
        __android_log_print(ANDROID_LOG_INFO, LogTag, "llama.cpp uses the Vulkan driver %s (%s)", LibraryName, Module->Name ? Module->Name : "unnamed");
        return true;
    }

    FVulkanDriver LoadDriver()
    {
        FVulkanDriver Driver;

        //LLAMA_VULKAN_DRIVER: unset = a Turnip driver packaged with the app if it supports this GPU,
        //"system" = the system driver, anything else = that driver library
        const char* Requested = getenv("LLAMA_VULKAN_DRIVER");
        const bool bRequested = Requested && Requested[0] != '\0';
        if (!bRequested || strcmp(Requested, "system") != 0)
        {
            if (TryHalDriver(bRequested ? Requested : "libvulkan_freedreno.so", bRequested, Driver))
            {
                return Driver;
            }
        }

        Driver.SystemLibrary = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
        if (Driver.SystemLibrary)
        {
            Driver.GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(Driver.SystemLibrary, "vkGetInstanceProcAddr"));
        }
        return Driver;
    }

    const FVulkanDriver& GetDriver()
    {
        static const FVulkanDriver Driver = LoadDriver();
        return Driver;
    }

    // The instance ggml-vulkan created (it creates one per process), for the functions it calls directly
    std::atomic<VkInstance> CreatedInstance{ VK_NULL_HANDLE };

    VKAPI_ATTR VkResult VKAPI_CALL CreateInstanceHook(const VkInstanceCreateInfo* CreateInfo, const VkAllocationCallbacks* Allocator, VkInstance* Instance)
    {
        const FVulkanDriver& Driver = GetDriver();
        PFN_vkCreateInstance Create = Driver.CreateInstance;
        if (!Create && Driver.GetInstanceProcAddr)
        {
            Create = reinterpret_cast<PFN_vkCreateInstance>(Driver.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
        }
        if (!Create)
        {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        const VkResult Result = Create(CreateInfo, Allocator, Instance);
        if (Result == VK_SUCCESS)
        {
            CreatedInstance = *Instance;
        }
        return Result;
    }

    // Resolves a directly called function once: from the driver through ggml's instance, else from libvulkan
    template <typename FunctionType>
    FunctionType Resolve(std::atomic<FunctionType>& Cache, const char* Name)
    {
        FunctionType Function = Cache.load(std::memory_order_acquire);
        if (Function)
        {
            return Function;
        }
        const FVulkanDriver& Driver = GetDriver();
        const VkInstance Instance = CreatedInstance.load();
        if (Instance && Driver.GetInstanceProcAddr)
        {
            Function = reinterpret_cast<FunctionType>(Driver.GetInstanceProcAddr(Instance, Name));
        }
        if (!Function && Driver.SystemLibrary)
        {
            Function = reinterpret_cast<FunctionType>(dlsym(Driver.SystemLibrary, Name));
        }
        if (Function)
        {
            Cache.store(Function, std::memory_order_release);
        }
        return Function;
    }
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance Instance, const char* Name)
{
    const FVulkanDriver& Driver = GetDriver();
    if (!Driver.GetInstanceProcAddr || !Name)
    {
        return nullptr;
    }
    if (strcmp(Name, "vkCreateInstance") == 0)
    {
        return reinterpret_cast<PFN_vkVoidFunction>(&CreateInstanceHook);
    }
    if (strcmp(Name, "vkGetInstanceProcAddr") == 0)
    {
        return reinterpret_cast<PFN_vkVoidFunction>(&vkGetInstanceProcAddr);
    }
    return Driver.GetInstanceProcAddr(Instance, Name);
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice Device, const char* Name)
{
    static std::atomic<PFN_vkGetDeviceProcAddr> Cache{ nullptr };
    const PFN_vkGetDeviceProcAddr Real = Resolve(Cache, "vkGetDeviceProcAddr");
    return Real ? Real(Device, Name) : nullptr;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice PhysicalDevice, VkPhysicalDeviceFeatures2* Features)
{
    static std::atomic<PFN_vkGetPhysicalDeviceFeatures2> Cache{ nullptr };
    const PFN_vkGetPhysicalDeviceFeatures2 Real = Resolve(Cache, "vkGetPhysicalDeviceFeatures2");
    if (Real)
    {
        Real(PhysicalDevice, Features);
    }
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer CommandBuffer, VkBuffer SrcBuffer, VkBuffer DstBuffer, uint32_t RegionCount, const VkBufferCopy* Regions)
{
    static std::atomic<PFN_vkCmdCopyBuffer> Cache{ nullptr };
    const PFN_vkCmdCopyBuffer Real = Resolve(Cache, "vkCmdCopyBuffer");
    if (Real)
    {
        Real(CommandBuffer, SrcBuffer, DstBuffer, RegionCount, Regions);
    }
}

#endif // PLATFORM_ANDROID
