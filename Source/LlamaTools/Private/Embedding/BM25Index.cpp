// Copyright 2025-current Getnamo.

#include "Embedding/BM25Index.h"
#include "LlamaUtility.h"

namespace
{
    static const TSet<FString>& AsciiStopwords()
    {
        static const TSet<FString> Words = {
            TEXT("a"), TEXT("an"), TEXT("the"), TEXT("and"), TEXT("or"), TEXT("but"),
            TEXT("of"), TEXT("in"), TEXT("on"), TEXT("at"), TEXT("to"), TEXT("for"),
            TEXT("with"), TEXT("by"), TEXT("from"), TEXT("as"), TEXT("is"), TEXT("are"),
            TEXT("was"), TEXT("were"), TEXT("be"), TEXT("been"), TEXT("being"),
            TEXT("have"), TEXT("has"), TEXT("had"), TEXT("do"), TEXT("does"), TEXT("did"),
            TEXT("it"), TEXT("its"), TEXT("this"), TEXT("that"), TEXT("these"), TEXT("those"),
            TEXT("i"), TEXT("you"), TEXT("he"), TEXT("she"), TEXT("we"), TEXT("they"),
            TEXT("not"), TEXT("no"), TEXT("so"), TEXT("if"), TEXT("then"), TEXT("than")
        };
        return Words;
    }

    static bool IsTokenChar(TCHAR Ch)
    {
        return FChar::IsAlnum(Ch) || Ch == TEXT('_');
    }
}

void FBM25Index::Tokenize(const FString& Text, TArray<FString>& OutTokens, bool bFilterStopwords)
{
    OutTokens.Reset();
    const int32 Len = Text.Len();
    int32 Cursor = 0;
    FString Buf;
    Buf.Reserve(32);

    while (Cursor < Len)
    {
        const TCHAR Ch = Text[Cursor];
        if (IsTokenChar(Ch))
        {
            Buf.AppendChar(FChar::ToLower(Ch));
        }
        else if (Buf.Len() > 0)
        {
            if (Buf.Len() >= 2 && (!bFilterStopwords || !AsciiStopwords().Contains(Buf)))
            {
                OutTokens.Add(MoveTemp(Buf));
            }
            Buf.Reset();
        }
        ++Cursor;
    }
    if (Buf.Len() >= 2 && (!bFilterStopwords || !AsciiStopwords().Contains(Buf)))
    {
        OutTokens.Add(MoveTemp(Buf));
    }
}

FBM25Index::FBM25Index() = default;
FBM25Index::~FBM25Index() = default;

void FBM25Index::Reset()
{
    Postings.Empty();
    DocLengths.Empty();
    Idf.Empty();
    AvgDocLen = 0.f;
    bFinalized = false;
}

int32 FBM25Index::NumDocuments() const
{
    return DocLengths.Num();
}

void FBM25Index::AddDocument(int64 DocId, const FString& Text)
{
    TArray<FString> Tokens;
    Tokenize(Text, Tokens, Params.bFilterStopwords);

    // Aggregate term frequencies for this doc.
    TMap<FString, uint16> TermFreq;
    TermFreq.Reserve(Tokens.Num());
    for (const FString& Tok : Tokens)
    {
        uint16& Cnt = TermFreq.FindOrAdd(Tok, 0);
        if (Cnt < TNumericLimits<uint16>::Max()) { ++Cnt; }
    }

    // If the doc already exists, drop its previous postings first.
    if (DocLengths.Contains(DocId))
    {
        for (auto& Pair : Postings)
        {
            Pair.Value.RemoveAll([DocId](const TPair<int64, uint16>& E){ return E.Key == DocId; });
        }
    }
    DocLengths.Add(DocId, static_cast<uint32>(Tokens.Num()));

    for (const auto& KV : TermFreq)
    {
        TArray<TPair<int64, uint16>>& List = Postings.FindOrAdd(KV.Key);
        List.Emplace(DocId, KV.Value);
    }

    bFinalized = false;
}

void FBM25Index::Finalize()
{
    Idf.Empty();

    const int32 N = DocLengths.Num();
    if (N == 0) { AvgDocLen = 0.f; bFinalized = true; return; }

    int64 TotalLen = 0;
    for (const auto& KV : DocLengths) { TotalLen += KV.Value; }
    AvgDocLen = static_cast<float>(static_cast<double>(TotalLen) / static_cast<double>(N));

    Idf.Reserve(Postings.Num());
    for (const auto& KV : Postings)
    {
        const int32 Df = KV.Value.Num();
        // BM25+ IDF (always positive)
        const float Numer = static_cast<float>(N) - static_cast<float>(Df) + 0.5f;
        const float Denom = static_cast<float>(Df) + 0.5f;
        const float Val = FMath::Loge((Numer / Denom) + 1.f);
        Idf.Add(KV.Key, Val);
    }
    bFinalized = true;
}

void FBM25Index::Query(const FString& QueryText, int32 K,
                       TArray<int64>& OutIds, TArray<float>& OutScores) const
{
    OutIds.Reset();
    OutScores.Reset();

    if (!bFinalized || K <= 0 || DocLengths.Num() == 0) { return; }

    TArray<FString> QueryTokens;
    Tokenize(QueryText, QueryTokens, Params.bFilterStopwords);
    if (QueryTokens.Num() == 0) { return; }

    TMap<int64, float> Scores;
    Scores.Reserve(64);

    const float K1 = Params.K1;
    const float B  = Params.B;

    for (const FString& Term : QueryTokens)
    {
        const TArray<TPair<int64, uint16>>* PostingList = Postings.Find(Term);
        if (!PostingList) { continue; }
        const float TermIdf = Idf.FindRef(Term);
        if (TermIdf <= 0.f) { continue; }

        for (const TPair<int64, uint16>& Entry : *PostingList)
        {
            const int64  DocId = Entry.Key;
            const float  Tf    = static_cast<float>(Entry.Value);
            const uint32 DocLen = DocLengths.FindRef(DocId);
            const float  LengthNorm = K1 * (1.f - B + B * (static_cast<float>(DocLen) / FMath::Max(AvgDocLen, 1.f)));
            const float  Contribution = TermIdf * (Tf * (K1 + 1.f)) / (Tf + LengthNorm);
            float& Acc = Scores.FindOrAdd(DocId, 0.f);
            Acc += Contribution;
        }
    }

    // Top-K via partial sort.
    OutIds.Reserve(Scores.Num());
    OutScores.Reserve(Scores.Num());
    for (const auto& KV : Scores)
    {
        OutIds.Add(KV.Key);
        OutScores.Add(KV.Value);
    }

    // Sort indices by descending score.
    TArray<int32> Indices;
    Indices.SetNumUninitialized(OutIds.Num());
    for (int32 i = 0; i < Indices.Num(); ++i) { Indices[i] = i; }
    Indices.Sort([&OutScores](int32 A, int32 B){ return OutScores[A] > OutScores[B]; });

    const int32 Take = FMath::Min(K, Indices.Num());
    TArray<int64> SortedIds;  SortedIds.Reserve(Take);
    TArray<float> SortedScores; SortedScores.Reserve(Take);
    for (int32 i = 0; i < Take; ++i)
    {
        SortedIds.Add(OutIds[Indices[i]]);
        SortedScores.Add(OutScores[Indices[i]]);
    }
    OutIds = MoveTemp(SortedIds);
    OutScores = MoveTemp(SortedScores);
}

bool FBM25Index::Save(FArchive& Ar)
{
    if (!bFinalized) { Finalize(); }

    uint32 Magic = 0x424D3235; // 'BM25'
    uint32 Version = 1;
    Ar << Magic;
    Ar << Version;

    Ar << Params.K1 << Params.B << Params.bFilterStopwords;
    Ar << AvgDocLen;

    int32 NDocs = DocLengths.Num();
    Ar << NDocs;
    for (auto& KV : DocLengths)
    {
        int64 K = KV.Key; uint32 V = KV.Value;
        Ar << K << V;
    }

    int32 NPostings = Postings.Num();
    Ar << NPostings;
    for (auto& KV : Postings)
    {
        FString Term = KV.Key;
        Ar << Term;
        int32 Count = KV.Value.Num();
        Ar << Count;
        for (auto& Entry : KV.Value)
        {
            int64 D = Entry.Key; uint16 T = Entry.Value;
            Ar << D << T;
        }
    }

    int32 NIdf = Idf.Num();
    Ar << NIdf;
    for (auto& KV : Idf)
    {
        FString Term = KV.Key; float V = KV.Value;
        Ar << Term << V;
    }

    return true;
}

bool FBM25Index::Load(FArchive& Ar)
{
    Reset();

    uint32 Magic = 0, Version = 0;
    Ar << Magic;
    Ar << Version;
    if (Magic != 0x424D3235 || Version != 1) { return false; }

    Ar << Params.K1 << Params.B << Params.bFilterStopwords;
    Ar << AvgDocLen;

    int32 NDocs = 0;
    Ar << NDocs;
    DocLengths.Reserve(NDocs);
    for (int32 i = 0; i < NDocs; ++i)
    {
        int64 K = 0; uint32 V = 0;
        Ar << K << V;
        DocLengths.Add(K, V);
    }

    int32 NPostings = 0;
    Ar << NPostings;
    Postings.Reserve(NPostings);
    for (int32 i = 0; i < NPostings; ++i)
    {
        FString Term;
        Ar << Term;
        int32 Count = 0;
        Ar << Count;
        TArray<TPair<int64, uint16>> List;
        List.Reserve(Count);
        for (int32 j = 0; j < Count; ++j)
        {
            int64 D = 0; uint16 T = 0;
            Ar << D << T;
            List.Emplace(D, T);
        }
        Postings.Add(Term, MoveTemp(List));
    }

    int32 NIdf = 0;
    Ar << NIdf;
    Idf.Reserve(NIdf);
    for (int32 i = 0; i < NIdf; ++i)
    {
        FString Term; float V = 0.f;
        Ar << Term << V;
        Idf.Add(Term, V);
    }

    bFinalized = true;
    return true;
}
