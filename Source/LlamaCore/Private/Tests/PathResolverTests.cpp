// Copyright 2025-current Getnamo.

#include "Misc/AutomationTest.h"

#if WITH_DEV_AUTOMATION_TESTS

#include "LlamaUtility.h"
#include "Misc/Paths.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FLlamaParsePathIntoFullPathTest,
    "LlamaCore.Paths.ParsePathIntoFullPath",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FLlamaParsePathIntoFullPathTest::RunTest(const FString& /*Parameters*/)
{
    const FString ModelsRoot = FPaths::ConvertRelativePathToFull(FLlamaPaths::ModelsRelativeRootPath());

    // 1) `./foo.gguf` — canonical Saved/Models-relative form. Must resolve under the models root.
    {
        const FString R = FLlamaPaths::ParsePathIntoFullPath(TEXT("./foo.gguf"));
        TestTrue(FString::Printf(TEXT("./ form resolves under models root (got: %s, root: %s)"), *R, *ModelsRoot),
            R.StartsWith(ModelsRoot));
        TestTrue(TEXT("./ form ends in foo.gguf"), R.EndsWith(TEXT("foo.gguf")));
    }

    // 2) `.\foo.gguf` — Windows-style separator variant.
    {
        const FString R = FLlamaPaths::ParsePathIntoFullPath(TEXT(".\\foo.gguf"));
        TestTrue(TEXT(".\\ form resolves under models root"), R.StartsWith(ModelsRoot));
    }

    // 3) `../../something/foo.gguf` — CWD-relative path that happens to start with `.`.
    //    Must NOT be prefixed by ModelsRoot. UE's ConvertRelativePathToFull will resolve
    //    it against CWD, so the result should not contain `..` and should not begin with
    //    the Saved/Models segment.
    {
        const FString R = FLlamaPaths::ParsePathIntoFullPath(TEXT("../../foo.gguf"));
        TestFalse(FString::Printf(TEXT("../ form must NOT be treated as models-relative (got: %s)"), *R),
            R.StartsWith(ModelsRoot));
        TestFalse(TEXT("../ form fully resolved (no leftover ..)"), R.Contains(TEXT("..")));
    }

    // 4) Bare relative path with no leading `.` — also must not be prefixed.
    {
        const FString R = FLlamaPaths::ParsePathIntoFullPath(TEXT("foo.gguf"));
        TestFalse(TEXT("plain relative not models-prefixed"), R.StartsWith(ModelsRoot));
    }

    // 5) Absolute path — passes through unchanged (modulo full-path normalization).
    {
        const FString Abs = TEXT("C:/somewhere/else/model.gguf");
        const FString R = FLlamaPaths::ParsePathIntoFullPath(Abs);
        TestTrue(TEXT("absolute path retains its drive/root"), R.StartsWith(TEXT("C:/")));
        TestTrue(TEXT("absolute path retains filename"), R.EndsWith(TEXT("model.gguf")));
        TestFalse(TEXT("absolute path not models-prefixed"), R.StartsWith(ModelsRoot));
    }

    return true;
}

#endif // WITH_DEV_AUTOMATION_TESTS
