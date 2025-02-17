#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <deque>
#include <mutex>
#include "CoreMinimal.h"
#include "FLLMModelParams.h"
#include "llama.h"
#include "Q.h"



namespace Internal
{
    class FLlama
    {
    public:
        bool bShouldLog = true;

        FLlama();
        ~FLlama();

        void StartStopThread(bool bShouldRun);

        void Activate(bool bReset, const FLLMModelParams& InputParams);
        void Deactivate();
        void InsertPrompt(FString Prompt);
        void Process();
        void StopGenerating();
        void ResumeGenerating();

        void UpdateParams(const FLLMModelParams& InputParams);
        
        FLLMModelParams Params;

        std::function<void(FString, int32)> OnTokenCb;
        std::function<void(bool, float)> OnEosCb;
        std::function<void(void)> OnStartEvalCb;
        std::function<void(void)> OnContextResetCb;
        std::function<void(FString)> OnErrorCb;

        static FString ModelsRelativeRootPath();
        static FString ParsePathIntoFullPath(const FString& InRelativeOrAbsolutePath);
        std::vector<llama_token> my_llama_tokenize(
            llama_context* InputContext,
            const std::string& Text,
            std::vector<llama_token>& InputRes,
            bool AddBos
        );

    private:
        llama_model* Model = nullptr;
        llama_context* Context = nullptr;

        Q qMainToThread;
        Q qThreadToMain;

        std::atomic_bool bRunning = false;
        std::thread qThread;

        std::vector<std::vector<llama_token>> StopSequences;
        std::vector<llama_token> EmbdInput;
        std::vector<llama_token> Embd;
        std::vector<llama_token> Res;
        int NPast = 0;
        std::vector<llama_token> LastNTokens;
        int NConsumed = 0;
        bool Eos = false;
        bool bStartedEvalLoop = false;
        double StartEvalTime = 0.f;
        int32 StartContextLength = 0;

        void ThreadRun();
        void UnsafeActivate(bool bReset);
        void UnsafeDeactivate();
        void UnsafeInsertPrompt(FString);
        bool HasEnding(std::string const& FullString, std::string const& Ending);
        void EmitErrorMessage(const FString& ErrorMessage, bool bLogErrorMessage = true);
    };
}
