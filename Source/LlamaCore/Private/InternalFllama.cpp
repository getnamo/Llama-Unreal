#include "InternalFLlama.h"
#include "LlamaComponent.h"
#include "HAL/PlatformTime.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <deque>
#include <functional>
#include <vector>
#include <string>

namespace Internal {

FLlama::FLlama() {
    // No thread startup unless initialized
}

FLlama::~FLlama() {
    bRunning = false;
    if (qThread.joinable()) {
        qThread.join();
    }
}

void FLlama::StartStopThread(bool bShouldRun) {
    if (bShouldRun) {
        if (bRunning) {
            return;
        }
        bRunning = true;
        qThread = std::thread([this]() {
            ThreadRun();
        });
    } else {
        bRunning = false;
        if (qThread.joinable()) {
            qThread.join();
        }
    }
}

void FLlama::Activate(bool bReset, const FLLMModelParams& InputParams) {
    // Crear una copia explícita de Params
    FLLMModelParams SafeParams = InputParams;

    qMainToThread.Enqueue([this, bReset, SafeParams = std::move(SafeParams)]() mutable {
        this->Params = SafeParams; // Asegúrate de que Params esté definido correctamente
        UnsafeActivate(bReset);
    });
}

//void FLlama::Activate(bool bReset, const FLLMModelParams& Params) {
//    FLLMModelParams SafeParams = Params;
//    qMainToThread.Enqueue([this, bReset, SafeParams]() mutable {
//        Params = SafeParams;
//        UnsafeActivate(bReset);
//    });
//}

void FLlama::Deactivate() {
    qMainToThread.Enqueue([this]() {
        UnsafeDeactivate();
    });
}

void FLlama::InsertPrompt(FString v) {
    qMainToThread.Enqueue([this, v = std::move(v)]() mutable {
        UnsafeInsertPrompt(std::move(v));
    });
}

void FLlama::UnsafeInsertPrompt(FString v) {
    if (!Context) {
        UE_LOG(LogTemp, Error, TEXT("Llama not activated"));
        return;
    }
    std::string stdV = std::string(" ") + TCHAR_TO_UTF8(*v);
    std::vector<llama_token> lineInp = my_llama_tokenize(Context, stdV, Res, false);
    EmbdInput.insert(EmbdInput.end(), lineInp.begin(), lineInp.end());
}

void FLlama::Process() {
    while (qThreadToMain.ProcessQ());
}

void FLlama::StopGenerating() {
    qMainToThread.Enqueue([this]() {
        Eos = true;
    });
}

void FLlama::ResumeGenerating() {
    qMainToThread.Enqueue([this]() {
        Eos = false;
    });
}

void FLlama::UpdateParams(const FLLMModelParams& InputParams) {
    FLLMModelParams SafeParams = InputParams;
    qMainToThread.Enqueue([this, SafeParams]() mutable {
        this->Params = SafeParams;
    });
}

FString FLlama::ModelsRelativeRootPath() {
    FString AbsoluteFilePath;
#if PLATFORM_ANDROID
    AbsoluteFilePath = FPaths::Combine(FPaths::Combine(FString(FAndroidMisc::GamePersistentDownloadDir()), "Models/"));
#else
    AbsoluteFilePath = FPaths::ConvertRelativePathToFull(FPaths::Combine(FPaths::ProjectSavedDir(), "Models/"));
#endif
    return AbsoluteFilePath;
}

FString FLlama::ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath) {
    FString FinalPath;
    if (InRelativeOrAbsolutePath.StartsWith(TEXT("."))) {
        FinalPath = FPaths::ConvertRelativePathToFull(ModelsRelativeRootPath() + InRelativeOrAbsolutePath);
    } else {
        FinalPath = FPaths::ConvertRelativePathToFull(InRelativeOrAbsolutePath);
    }
    return FinalPath;
}

    std::vector<llama_token> FLlama::my_llama_tokenize(
        llama_context* InputContext,
        const std::string& Text,
        std::vector<llama_token>& InputRes,
        bool AddBos
    ) {
    UE_LOG(LogTemp, Warning, TEXT("Tokenize `%s`"), UTF8_TO_TCHAR(Text.c_str()));
    InputRes.resize(Text.size() + (int)AddBos);
    const int n = llama_tokenize(
        llama_get_model(InputContext),
        Text.c_str(),
        Text.length(),
        InputRes.data(),
        InputRes.size(),
        AddBos,
        false
    );
    InputRes.resize(n);
    return InputRes;
}

void FLlama::ThreadRun() {
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread is running"), this);
    while (bRunning) {
        while (qMainToThread.ProcessQ());

        if (!Model) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        if (Eos && (int)EmbdInput.size() <= NConsumed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }

        // Additional processing logic here
    }
    UnsafeDeactivate();
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread stopped"), this);
}

void FLlama::UnsafeActivate(bool bReset) {
    UE_LOG(LogTemp, Warning, TEXT("%p Loading LLM model %p bReset: %d"), this, Model, bReset);
    if (bReset) {
        UnsafeDeactivate();
    }
    if (Model) {
        return;
    }

    // Additional activation logic here
}

void FLlama::UnsafeDeactivate() {
    bStartedEvalLoop = false;
    StopSequences.clear();
    if (!Model) {
        return;
    }
    llama_free(Context);
    Context = nullptr;
    llama_free_model(Model);
    Model = nullptr;

    qThreadToMain.Enqueue([this]() {
        if (OnContextResetCb) {
            OnContextResetCb();
        }
    });
}

void FLlama::EmitErrorMessage(const FString& ErrorMessage, bool bLogErrorMessage) {
    if (bLogErrorMessage) {
        UE_LOG(LogTemp, Error, TEXT("%s"), *ErrorMessage);
    }
    qThreadToMain.Enqueue([this, ErrorMessage]() {
        if (OnErrorCb) {
            OnErrorCb(ErrorMessage);
        }
    });
}

bool FLlama::HasEnding(std::string const& FullString, std::string const& Ending) {
    return FullString.size() >= Ending.size() &&
           FullString.compare(FullString.size() - Ending.size(), Ending.size(), Ending) == 0;
}
    


} // namespace Internal
