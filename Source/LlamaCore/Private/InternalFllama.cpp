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

#include "common/common.h"

using namespace std;

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
    FLLMModelParams SafeParams = InputParams;

    qMainToThread.Enqueue([this, bReset, SafeParams = std::move(SafeParams)]() mutable {
        this->Params = SafeParams; // Asegúrate de que Params esté definido correctamente
        UnsafeActivate(bReset);
    });
}
    

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

void FLlama::ThreadRun()
{
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread is running"), this);
    const int NPredict = -1;
    const int NKeep = 0;
    const int NBatch = Params.BatchCount;

    while (bRunning)
    {
        while (qMainToThread.ProcessQ())
            ;
        if (!Model)
        {
            using namespace chrono_literals;
            this_thread::sleep_for(200ms);
            continue;
        }

        if (Eos && (int)EmbdInput.size() <= NConsumed)
        {
            using namespace chrono_literals;
            this_thread::sleep_for(200ms);
            continue;
        }
        if (Eos == false && !bStartedEvalLoop)
        {
            bStartedEvalLoop = true;
            StartEvalTime = FPlatformTime::Seconds();
            StartContextLength = NPast; //(int32)LastNTokens.size(); //(int32)embd_inp.size();

            qThreadToMain.Enqueue([this] 
            {    
                if (!OnStartEvalCb)
                {
                    return;
                }
                OnStartEvalCb();
            });
        }


        Eos = false;

        const int NCtx = llama_n_ctx(Context);
        if (Embd.size() > 0)
        {
            // Note: NCtx - 4 here is to match the logic for commandline Prompt handling via
            // --Prompt or --file which uses the same value.
            int MaxEmbdSize = NCtx - 4;
            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int)Embd.size() > MaxEmbdSize)
            {
                uint64 SkippedTokens = Embd.size() - MaxEmbdSize;
                FString ErrorMsg = FString::Printf(TEXT("<<input too long: skipped %zu token%s>>"), 
                    SkippedTokens,
                    SkippedTokens != 1 ? "s" : "");
                EmitErrorMessage(ErrorMsg);
                Embd.resize(MaxEmbdSize);
            }

            // infinite Text generation via context swapping
            // if we run out of context:
            // - take the NKeep first tokens from the original Prompt (via NPast)
            // - take half of the last (NCtx - NKeep) tokens and recompute the logits in batches
            if (NPast + (int)Embd.size() > NCtx)
            {
                UE_LOG(LogTemp, Warning, TEXT("%p context resetting"), this);
                if (NPredict == -2)
                {
                    FString ErrorMsg = TEXT("context full, stopping generation");
                    EmitErrorMessage(ErrorMsg);
                    UnsafeDeactivate();
                    continue;
                }

                const int NLeft = NPast - NKeep;
                // always keep the first token - BOS
                NPast = max(1, NKeep);

                // insert NLeft/2 tokens at the start of embd from LastNTokens
                Embd.insert(Embd.begin(),
                                        LastNTokens.begin() + NCtx - NLeft / 2 - Embd.size(),
                                        LastNTokens.end() - Embd.size());
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            for (int i = 0; i < (int)Embd.size(); i += NBatch)
            {

                int NEval = (int)Embd.size() - i;
                if (NEval > NBatch)
                {
                    NEval = NBatch;
                }
                

                if (bShouldLog)
                {
                    string Str = string{};
                    for (auto j = 0; j < NEval; ++j)
                    {
                        Str += llama_detokenize(Context, { Embd[i + j] });
                    }
                    UE_LOG(LogTemp, Warning, TEXT("%p eval tokens `%s`"), this, UTF8_TO_TCHAR(Str.c_str()));
                }

                if (llama_decode(Context, llama_batch_get_one(&Embd[i], NEval, NPast, 0)))
                {
                    FString ErrorMsg = TEXT("failed to eval");
                    EmitErrorMessage(ErrorMsg);
                    UnsafeDeactivate();
                    continue;
                }
                NPast += NEval;
            }

        }

        Embd.clear();

        bool bHaveHumanTokens = false;
        const FLLMModelAdvancedParams& P = Params.Advanced;

        if ((int)EmbdInput.size() <= NConsumed)
        {
            llama_token ID = 0;

            {
                float* Logits = llama_get_logits(Context);
                int NVocab = llama_n_vocab(llama_get_model(Context));

                vector<llama_token_data> Candidates;
                Candidates.reserve(NVocab);
                for (llama_token TokenID = 0; TokenID < NVocab; TokenID++)
                {
                    Candidates.emplace_back(llama_token_data{TokenID, Logits[TokenID], 0.0f});
                }

                llama_token_data_array CandidatesP = {Candidates.data(), Candidates.size(), false};

                // Apply penalties
                float NLLogit = Logits[llama_token_nl(llama_get_model(Context))];
                int LastNRepeat = min(min((int)LastNTokens.size(), P.RepeatLastN), NCtx);

                llama_sample_repetition_penalties(  Context,
                                                    &CandidatesP,
                                                    LastNTokens.data() + LastNTokens.size() - LastNRepeat,
                                                    LastNRepeat,
                                                    P.RepeatPenalty,
                                                    P.AlphaFrequency,
                                                    P.AlphaPresence);
                if (!P.PenalizeNl)
                {
                    Logits[llama_token_nl(llama_get_model(Context))] = NLLogit;
                }

                if (P.Temp <= 0)
                {
                    // Greedy sampling
                    ID = llama_sample_token_greedy(Context, &CandidatesP);
                }
                else
                {
                    if (P.Mirostat == 1)
                    {
                        static float MirostatMu = 2.0f * P.MirostatTau;
                        llama_sample_temp(Context, &CandidatesP, P.Temp);
                        ID = llama_sample_token_mirostat(
                            Context, &CandidatesP, P.MirostatTau, P.MirostatEta, P.MirostatM, &MirostatMu);
                    }
                    else if (P.Mirostat == 2)
                    {
                        static float MirostatMu = 2.0f * P.MirostatTau;
                        llama_sample_temp(Context, &CandidatesP, P.Temp);
                        ID = llama_sample_token_mirostat_v2(
                            Context, &CandidatesP, P.MirostatTau, P.MirostatEta, &MirostatMu);
                    }
                    else
                    {
                        // Temperature sampling
                        llama_sample_top_k(Context, &CandidatesP, P.TopK, 1);
                        llama_sample_tail_free(Context, &CandidatesP, P.TfsZ, 1);
                        llama_sample_typical(Context, &CandidatesP, P.TypicalP, 1);
                        llama_sample_top_p(Context, &CandidatesP, P.TopP, 1);
                        llama_sample_temp(Context, &CandidatesP, P.Temp);
                        ID = llama_sample_token(Context, &CandidatesP);
                    }
                }

                LastNTokens.erase(LastNTokens.begin());
                LastNTokens.push_back(ID);
            }

            // add it to the context
            Embd.push_back(ID);
        }
        else
        {
            // some user input remains from Prompt or interaction, forward it to processing
            while ((int)EmbdInput.size() > NConsumed)
            {
                const int tokenId = EmbdInput[NConsumed];
                Embd.push_back(tokenId);
                LastNTokens.erase(LastNTokens.begin());
                LastNTokens.push_back(EmbdInput[NConsumed]);
                bHaveHumanTokens = true;
                ++NConsumed;
                if ((int)Embd.size() >= NBatch)
                {
                    break;
                }
            }
        }

        // TODO: Revert these changes to the commented code when the llama.cpp add the llama_detokenize function.
        
        // display Text
        // for (auto Id : embd)
        // {
        //     FString token = llama_detokenize(Context, Id);
        //     qThreadToMain.Enqueue([token = move(token), this]() {
        //         if (!OnTokenCb)
        //             return;
        //         OnTokenCb(move(token));
        //     });
        // }
        
        FString Token = UTF8_TO_TCHAR(llama_detokenize(Context, Embd).c_str());

        //Debug block
        //NB: appears full history is not being input back to the model,
        // does Llama not need input copying for proper context?
        //FString history1 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, embd_inp).c_str()); 
        //FString history2 = UTF8_TO_TCHAR(llama_detokenize_bpe(Context, LastNTokens).c_str());
        //UE_LOG(LogTemp, Log, TEXT("history1: %s, history2: %s"), *history1, *history2);
        int32 NewContextLength = NPast; //(int32)LastNTokens.size();

        
        qThreadToMain.Enqueue([token = std::move(Token), NewContextLength,  this] {
            if (!OnTokenCb)
                return;
            OnTokenCb(std::move(token), NewContextLength);
        });
        ////////////////////////////////////////////////////////////////////////

            
        auto StringStopTest = [&]
        {
            if (StopSequences.empty())
                return false;
            if (bHaveHumanTokens)
                return false;                

            for (vector<llama_token> StopSeq : StopSequences)
            {
                FString Sequence = UTF8_TO_TCHAR(llama_detokenize(Context, StopSeq).c_str());
                Sequence = Sequence.TrimStartAndEnd();

                vector<llama_token> EndSeq;
                for (unsigned i = 0U; i < StopSeq.size(); ++i)
                {
                    EndSeq.push_back(LastNTokens[LastNTokens.size() - StopSeq.size() + i]);
                }
                FString EndString = UTF8_TO_TCHAR(llama_detokenize(Context, EndSeq).c_str());
                
                if (bShouldLog) 
                {
                    UE_LOG(LogTemp, Log, TEXT("stop vs end: #%s# vs #%s#"), *Sequence, *EndString);
                }
                if (EndString.Contains(Sequence))
                {
                    UE_LOG(LogTemp, Warning, TEXT("String match found, String EOS triggered."));
                    return true;
                }
                

                if (LastNTokens.size() < StopSeq.size())
                    return false;
                bool bMatch = true;
                for (unsigned i = 0U; i < StopSeq.size(); ++i)
                    if (LastNTokens[LastNTokens.size() - StopSeq.size() + i] != StopSeq[i])
                    {
                        bMatch = false;
                        break;
                    }
                if (bMatch)
                    return true;
            }
            return false;
        };

        bool EOSTriggered = false;
        bool bStandardTokenEOS = (!Embd.empty() && Embd.back() == llama_token_eos(llama_get_model(Context)));

        //check
        if (!bStandardTokenEOS)
        {
            EOSTriggered = StringStopTest();
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("%p Standard EOS triggered"), this);
            EOSTriggered = true;
        }

        if (EOSTriggered)
        {
            //UE_LOG(LogTemp, Warning, TEXT("%p EOS"), this);
            Eos = true;
            const bool StopSeqSafe = EOSTriggered;
            const int32 DeltaTokens = NewContextLength - StartContextLength;
            const double EosTime = FPlatformTime::Seconds();
            const float TokensPerSecond = double(DeltaTokens) / (EosTime - StartEvalTime);

            bStartedEvalLoop = false;
            

            //notify main thread we're done
            qThreadToMain.Enqueue([StopSeqSafe, TokensPerSecond, this] 
            {
                if (!OnEosCb)
                {
                    return;
                }
                OnEosCb(StopSeqSafe, TokensPerSecond);
            });
        }
    }
    UnsafeDeactivate();
    UE_LOG(LogTemp, Warning, TEXT("%p Llama thread stopped"), this);
}
    
void FLlama::UnsafeActivate(bool bReset)
{
    UE_LOG(LogTemp, Warning, TEXT("%p Loading LLM model %p bReset: %d"), this, Model, bReset);
    if (bReset)
        UnsafeDeactivate();
    if (Model)
        return;
    
    llama_context_params lparams = [this]()
    {
        llama_context_params lparams = llama_context_default_params();
        // -eps 1e-5 -t 8 -ngl 50
        lparams.n_ctx = Params.MaxContextLength;

        bool bIsRandomSeed = Params.Seed == -1;

        if(bIsRandomSeed){
            lparams.seed = time(nullptr);
        }
        else
        {
            lparams.seed = Params.Seed;
        }


        return lparams;
    }();

    llama_model_params mParams = llama_model_default_params();
    mParams.n_gpu_layers = Params.MaxContextLength;

    FString FullModelPath = ParsePathIntoFullPath(Params.PathToModel);

    UE_LOG(LogTemp, Log, TEXT("File at %s exists? %d"), *FullModelPath, FPaths::FileExists(FullModelPath));

    Model = llama_load_model_from_file(TCHAR_TO_UTF8(*FullModelPath), mParams);
    if (!Model)
    {
        FString ErrorMessage = FString::Printf(TEXT("%p unable to load model at %s"), this, *FullModelPath);

        EmitErrorMessage(ErrorMessage);
        UnsafeDeactivate();
        return;
    }

    //Read GGUF info
    gguf_ex_read_0(TCHAR_TO_UTF8(*FullModelPath));

    Context = llama_new_context_with_model(Model, lparams);
    NPast = 0;

    UE_LOG(LogTemp, Warning, TEXT("%p model context set to %p"), this, Context);

    // tokenize the Prompt
    string StdPrompt = string(" ") + TCHAR_TO_UTF8(*Params.Prompt);
    EmbdInput = my_llama_tokenize(Context, StdPrompt, Res, true /* add bos */);
    if (!Params.StopSequences.IsEmpty())
    {
        for (int i = 0; i < Params.StopSequences.Num(); ++i)
        {
            const FString& stopSeq = Params.StopSequences[i];
            string str = string{TCHAR_TO_UTF8(*stopSeq)};
            if (::isalnum(str[0]))
                str = " " + str;
            vector<llama_token> seq = my_llama_tokenize(Context, str, Res, false /* add bos */);
            StopSequences.emplace_back(std::move(seq));
        }
    }
    else
    {
        StopSequences.clear();
    }

    const int NCtx = llama_n_ctx(Context);

    if ((int)EmbdInput.size() > NCtx - 4)
    {
        FString ErrorMessage = FString::Printf(TEXT("prompt is too long (%d tokens, max %d)"), (int)EmbdInput.size(), NCtx - 4);
        EmitErrorMessage(ErrorMessage);
        UnsafeDeactivate();
        return;
    }

    // do one empty run to warm up the model
    llama_set_n_threads(Context, Params.Threads, Params.Threads);

    {
        vector<llama_token> Tmp = {
            llama_token_bos(llama_get_model(Context)),
        };
        llama_decode(Context, llama_batch_get_one(Tmp.data(), Tmp.size(), 0, 0));
        llama_reset_timings(Context);
    }
    LastNTokens.resize(NCtx);
    fill(LastNTokens.begin(), LastNTokens.end(), 0);
    NConsumed = 0;
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

}
