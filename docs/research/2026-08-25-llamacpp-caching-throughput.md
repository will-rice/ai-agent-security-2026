# llama.cpp caching/throughput for shared-prefix candidate replay (2026-08-25)

Context: the grader replays thousands of candidates through one resident GGUF
(`create_chat_completion`, temp=0, KV NOT reset). Each candidate = fixed [system+tool
schema] prefix + a user message differing only in a trailing host token + a ~28-token
tool-call decode. Score = candidates completed in a fixed budget.

## Confirmed mechanism (matches our board evidence)
- **Prefix reuse is a strict, exact TOKEN-prefix match** (`get_common_prefix`, server
  `cache_prompt=true` default). First differing token truncates all reuse.
- llama-cpp-python `create_completion`/`create_chat_completion` reuses the previous
  call's KV automatically (`longest_token_prefix`); `reset()` keeps native KV. So the
  long [system+schema+fixed-instruction] prefix is prefilled ~ONCE and amortized; only
  the host token onward + the ~28 decode tokens are paid per candidate.
- => url-LAST (host = last token) maximizes the shared prefix. This IS the mechanism and
  we already max it. Board evidence agrees (url-last +6; host-mid -40%).

## Levers that exist but are GRADER-SERVER CONFIG (we cannot set them)
- Slot save/restore (`--slot-save-path`, `/slots ?action=save|restore`), host-RAM prompt
  cache (`-cram/--cache-ram`, `--cache-idle-slots`): prefill the fixed prefix once and
  restore per candidate. We don't control the grader's flags.
- `--cache-reuse N`: reuse non-contiguous cached chunks after a mid-prompt divergence
  (KV shift). Irrelevant for pure host-last.
- Continuous batching (`-cb`, `-np`): packs decode across candidates. Server-side.

## THE decode lever (server-side, but reframes our search): n-gram / lookup speculation
- `--spec-type ngram-mod` / `-lcd,--lookup-cache-dynamic`: llama.cpp drafts the next
  tokens from an n-gram pool built from history and verifies in ONE forward pass. NO
  draft model needed. **`ngram-mod`'s hash pool is shared across all server slots -> it
  carries n-grams ACROSS candidates.** So once the common tool-call scaffolding is
  generated once, every later candidate DRAFTS it for free; only the novel host token is
  a "hard" decode. Greedy (temp=0, which the grader uses) => near-100% acceptance.
- IF the grader has this on: decode-token COUNT barely matters (drafted); only the number
  of UNPREDICTABLE tokens matters -- for us that is the single host token. So the champion
  is already near-optimal and chasing sub-28 decode is pointless (drafted anyway). If it
  is OFF: decode = full 28 tokens and leanness matters.
- This is the key UNKNOWN. It cannot be set by us; it can only be inferred from the board.

## Gotchas that BREAK prefix caching (keep our shapes clean)
- Any differing token EARLY in the prompt truncates reuse (no host-in-the-middle -- our
  url-last gate enforces this).
- BOS/add_special inconsistency, chat-template re-render injecting a date/reorder, or
  changing model/n_ctx/rope/sampler mid-stream all wipe reuse. The [system+schema] is
  grader-fixed; our message must not break the prefix (url-last does not).
- Greedy keeps the output constant => maximal ngram-speculation acceptance. Uniform pools
  (all candidates identical except the host) maximize the shared n-gram pool warmth.

## Bottom line for the candidate side
url-last + single-token host + uniform pool = near the caching-exploitation ceiling we
control. The powerful extra caching (slot save, cache-ram, ngram speculation) is the
grader's server config, not ours. If the grader speculative-decodes, our decode is already
near-free (only the host is hard) -- so the 137 leader is NOT winning on decode-token count.

Sources: llama.cpp server-context.cpp/server-task.h/README (context7 /ggml-org/llama.cpp);
docs/speculative.md; llama-cpp-python state/caching (DeepWiki 4.6); disc #13606.
