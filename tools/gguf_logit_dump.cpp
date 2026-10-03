// Dump the full logit matrix of a GGUF model for a fixed token window.
//
// Verification tool for audit §34–§40: the GGUF runtime check must compare *the
// same computation*, not an aggregate. Given a file of token ids (one per line),
// this writes float32 logits for every position as a raw matrix
// (n_pos x n_vocab, row-major, little-endian) so scripts/verify_gguf_logits.py can
// compare them against the native TinyMe engine without any statistics in
// between. A wrong RoPE element pairing shows up as a collapsed correlation and
// a near-random argmax agreement; a correct pairing shows up as f16-vs-f32 noise.
//
// Build:  see scripts/verify_gguf_logits.py (it compiles this file).
// Usage:  gguf_logit_dump MODEL.gguf TOKENS.txt LOGITS.bin [-c N] [--no-bos]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<llama_token> read_tokens(const char * path, bool add_bos) {
    std::ifstream in(path);
    std::vector<llama_token> tokens;
    if (add_bos) {
        tokens.push_back(1);   // <|bos|> of this vocabulary
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        tokens.push_back((llama_token) std::stoi(line));
    }
    return tokens;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf TOKENS.txt LOGITS.bin [-c N] [--no-bos]\n", argv[0]);
        return 2;
    }
    const char * model_path  = argv[1];
    const char * tokens_path = argv[2];
    const char * out_path    = argv[3];
    int  n_ctx   = 512;
    bool add_bos = true;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc) { n_ctx = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--no-bos"))      { add_bos = false; }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t) n_ctx;
    cparams.n_batch   = (uint32_t) n_ctx;
    cparams.n_threads = 2;
    cparams.no_perf   = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const std::vector<llama_token> tokens = read_tokens(tokens_path, add_bos);
    if (tokens.empty() || (int) tokens.size() > n_ctx) {
        fprintf(stderr, "token window must contain 1..%d ids (got %zu)\n", n_ctx, tokens.size());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // A plain batch only produces logits for its last token, so every position is
    // marked as needing logits explicitly.
    const int32_t n_tokens = (int32_t) tokens.size();
    std::vector<llama_token>     batch_tokens(tokens.begin(), tokens.end());
    std::vector<llama_pos>       batch_pos(n_tokens);
    std::vector<int32_t>         batch_nseq(n_tokens, 1);
    std::vector<int8_t>          batch_logits(n_tokens, 1);
    for (int32_t i = 0; i < n_tokens; i++) { batch_pos[i] = i; }
    llama_batch batch = {
        /*n_tokens =*/ n_tokens,
        /*token    =*/ batch_tokens.data(),
        /*embd     =*/ nullptr,
        /*pos      =*/ batch_pos.data(),
        /*n_seq_id =*/ batch_nseq.data(),
        /*seq_id   =*/ nullptr,
        /*logits   =*/ batch_logits.data(),
    };
    std::vector<llama_seq_id> seq_ids0(n_tokens, 0);
    std::vector<llama_seq_id *> seq_ptrs(n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) { seq_ptrs[i] = &seq_ids0[i]; }
    batch.seq_id = seq_ptrs.data();

    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }

    // llama_get_logits returns a contiguous matrix for the whole batch:
    // position p, vocabulary index v  ->  [p * n_vocab + v]
    const float * logits = llama_get_logits(ctx);
    if (!logits) { fprintf(stderr, "no logits\n"); return 1; }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    const uint32_t n_pos = (uint32_t) tokens.size();
    out.write((const char *) &n_pos, sizeof(n_pos));
    out.write((const char *) &n_vocab, sizeof(n_vocab));
    out.write((const char *) logits, (size_t) n_pos * (size_t) n_vocab * sizeof(float));
    out.close();

    printf("wrote %u positions x %d vocab = %llu floats (%s, bos=%s)\n", n_pos, n_vocab,
           (unsigned long long)((size_t) n_pos * (size_t) n_vocab), out_path,
           add_bos ? "yes" : "no");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
