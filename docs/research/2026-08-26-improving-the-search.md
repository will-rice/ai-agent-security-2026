# Improving the search — literature survey (2026-08-26)

Concrete, adoptable techniques for our optimizer, drawn from a 2023–2026 sweep of
LLM-as-optimizer, quality-diversity red-teaming, and black-box adversarial-prompt search.
Scoped to **our** setting, not generic jailbreak search.

## The reframing that matters most

Our problem is **not** the problem most red-team/jailbreak papers solve. PAIR, TAP, AutoDAN,
Rainbow Teaming, Ferret all fight to lift a **noisy binary "did it jailbreak"** signal (via an
LLM judge) on models where *firing is hard*. For us, firing is already solved; the real objective
is **minimise on-victim tokens of an already-firing prompt**, with a **cheap, exact, deterministic
reward** (real GGUF replay, greedy). That puts us far closer to **FunSearch-style program-golf**
than to jailbreak-ASR search. Two consequences drive everything:

1. **We under-exploit our biggest asset: exact deterministic reward.** It licenses brute,
   non-semantic local search (token ablation, coordinate descent) the prompt-opt papers *can't*
   run because their reward is noisy. Our LLM proposer does semantic work when most remaining wins
   are at the **token/format** level.
2. **The plateau is a format-reversion wall, not a semantic one.** Greedy reverts the victim to its
   trained tool-call format. Semantic mutation ("say it more firmly") cannot cross that wall. We
   need operators that act at the token level, or that *discover* a leaner format is reachable at
   all before asking the proposer to induce it.

## Cluster 1 — LLM-as-optimizer / prompt optimization

1. **EvoPrompt's Differential-Evolution operator** (arXiv 2309.08532). Build a child from the
   *difference between two archive members* (`a + F·(b−c)`), not free-form rewriting of the
   incumbent. Directly breaks "proposer regenerates a paraphrase of the champion." *Adapt:* give the
   proposer champion + two structurally-different elites and instruct "apply the delta that makes B
   leaner than C to the champion."
2. **OPRO** (2309.03409) — we already use its trajectory idea; its known failure mode is exactly
   ours (single best-so-far → lexical neighbours). Fix from 2025 work: **population + bandit-guided
   operator selection** (UCB/Thompson over our mutation operators), keeping diversifying operators
   alive.
3. **PromptBreeder** (2309.16797) — evolve the *mutation prompts* too. Lightweight version: keep a
   pool of mutation instructions, promote the ones that produce archive-improving children.
4. **CAPO — Cost-Aware Prompt Optimization** (2504.16005) — makes length/cost a **first-class
   Pareto objective**, validating tokens-as-axis over tokens-as-tiebreak.
- *Skip:* LLM-guided Bayesian Optimization (LLAMBO, 2402.03921) — BO's value is sample efficiency
  under expensive/noisy reward; ours is cheap+exact, so a surrogate buys nothing and adds fragility.

## Cluster 2 — Quality-Diversity for red-team / prompt search

1. **Rainbow Teaming** (2402.16822) — our direct ancestor. The actionable lesson is **descriptor
   design**: we currently key cells on *outcomes* (throughput, severity), so two format-identical
   prompts share a cell and the search collapses. **Switch to structural/format descriptors**
   (`{has <|constrain|>json, args-quoted?, reply-suppressed?, forge-style, scaffold-token-count}`).
   MAP-Elites is then *forced* to keep an elite in the "unquoted-args" cell even when it scores
   worse — illuminating the exact lean cells greedy collapses away.
2. **RainbowPlus** (2504.15047) — multi-elite-per-cell archive; preserves near-ties single-elite
   MAP-Elites discards, giving DE/crossover more structurally-distinct parents.
3. **Ferret** (2408.10701) — generate N mutations per step, rank with a cheap scorer, keep the best.
   We *own* the perfect cheap scorer. *Adapt:* widen from 1 child to N=5–8 per parent, replay all,
   archive-insert all, advance the best.
4. **CMA-ME emitters** (1912.02400) — reset an emitter to a *random* elite (not the champion) on
   stagnation. The direct cure for "reliably rediscovers the champion."

## Cluster 3 — Black-box adversarial / jailbreak search

1. **AutoDAN-Turbo** (2410.05295) — the one jailbreak paper that transfers. Core = a **lifelong
   strategy library**: discover reusable strategies from own successes, retrieve + compose them.
   *Adapt to token-min:* a **lever library** — named reusable token-savers (scheme-prefix short
   host, reply-suppression tail, analysis-channel forge, drop-final-message). Feed the proposer
   *retrieved levers as composable building blocks*, turning "mutate a monolith" into "compose known
   savers" so independent savings **stack**. Our CLAUDE.md already enumerates several — formalise
   them as entries.
2. **AutoDAN genetic** (2310.04451) — word/token-level crossover + mutation. Steal the lexical
   operator set, drop the role-play objective.
3. **TAP / PAIR** (2312.02119 / 2310.08419) — prune-before-expand. Cheaply reject a child by
   structural-descriptor hash *before* replay if it collides with an existing elite's format.
4. **GPTFuzzer** (2309.10253) — UCB-over-seeds selection: a clean way to pick *which* elite to
   mutate next.
- *Skip the objectives:* firing isn't our bottleneck — take operators and selection machinery, not
  the harm-judge loss. White-box token attacks (GCG/BEAST/COLD) are out — need gradients we can't
  get through GGUF replay.

## Cluster 4 — Escaping proposer monoculture

1. **Novelty search on a dedicated island** — because reward is cheap, afford an island that selects
   *purely on behavioural novelty*, ignoring score (novelty = distance in structural-descriptor
   space to k-nearest elites). Manufactures unfamiliar formats; quality islands then exploit
   whatever fires. Textbook cure for a deceptive landscape.
2. **FunSearch island model + reset** (Nature 2023; ecosystem ShinkaEvolve/CodeEvolve 2024–25). The
   closest methodological match. Copy verbatim: (a) multiple islands with rare champion migration;
   (b) periodic hard reset of the worst half, reseeded from the best.
3. **Surprise/novelty bonus in the meta-prompt** — show the proposer a *banned-patterns* list
   (top n-grams/format-signatures from the archive) and require a structurally-disjoint child.
4. **Lexicase selection** (1905.09374) — select parents by a randomly-ordered sequence of
   sub-objectives (gemma-tokens, then gpt-tokens, then fire-reliability), preserving specialists a
   scalar objective kills. Matches our two-pool per-model structure.

## Cluster 5 — Minimising generated tokens / overriding the trained default format

1. **Deterministic token-ablation minimiser** — for any firing candidate, greedily try deletions →
   shorter-token substitutions → span merges, keep any edit that *still fires with ≤ tokens*, iterate
   to a fixed point. The black-box analog of a GCG coordinate step, on our prompt tokens, with exact
   reward. **[IMPLEMENTED — see below.]**
2. **Target-then-induce (reachability probe)** — before chasing a leaner *format* (drop
   `<|constrain|>json`, unquoted args), sample the *victim* at temp>0, best-of-N: if the lean format
   *ever* appears it's reachable (hill-climb the prompt toward it); if *never* across many samples,
   the trained prior is too strong — **stop chasing it**. Converts "hope the LLM discovers a novel
   induction" into a measured, decidable subproblem.
3. **Few-shot format priming + assistant-turn prefill** — length-control work (TALE 2412.18547,
   2504.14350) finds in-context exemplars of the desired-length output beat "be brief" instructions.
   Prefill more of the assistant turn: every token validly in the prefill is one the victim doesn't
   decode. Frontier: prefill *into the tool-call itself* as far as the harmony parser accepts.
4. **Reply-suppression as an explicit search axis** — make post-tool-call suffix length a MAP-Elites
   descriptor so suppression variants are explored systematically, not stumbled on.

## Top-5 to try next (payoff-to-effort)

1. **Token-ablation minimiser** as a post-pass on every firing candidate. Highest payoff, lowest
   effort, exploits the reward we waste. **DONE (2026-08-26).**
2. **Reachability probe** before chasing any novel-format target. Cheap, decisive; stops wasted
   proposer cycles on unreachable formats.
3. **Structural MAP-Elites descriptors** + multi-elite-per-cell — the real monoculture cure.
4. **FunSearch islands + stagnation resets + a pure-novelty island** — reset to *random* elites.
5. **Lever library + DE-difference proposer operator** — make independent savings stack.
- Runner-up: operator/seed bandits to auto-balance explore vs exploit.

## What probably will NOT transfer
- BO/surrogates (LLAMBO) — reward is cheap+exact.
- Jailbreak *objectives* (PAIR/TAP/AutoDAN/GPTFuzzer/Ferret harm-judge) — take operators, not loss.
- White-box token attacks (GCG/BEAST/COLD) — no gradients through GGUF.
- Soft-prompt/embedding QD — needs embedding-level access we lack.
- Generic "be brief" length control — greedy ignores it; only few-shot exemplars + prefill survive.

## Status in this repo

- **#1 token-ablation minimiser — IMPLEMENTED** as `jed_attack.campaign.ablate.minimize_shape`
  (greedy exact-reward deletion, robust-gated to the word-host fill probes), wired as a post-pass on
  the optimizer champion (`optimize_prompts._ablate_champion`, fires only on a frontier change).
  First run on the champions: gpt 53→51 tok, gemma 51→48 tok — **all input-token savings; gen stayed
  pinned at 28/29**, an independent confirmation of the decode floor. Whether the input savings move
  the board is the open decode-vs-prefill question (queued as a real-board test).
- **#5 lever library + DE-difference operator — IMPLEMENTED** in `prompts.toml` (hot-reloads, no
  restart): the proposer now stacks named token-savers (short scheme, host-last, reply-suppression,
  forge, punctuation/space trims) onto the incumbent and applies the *delta* between two elites,
  instead of paraphrasing the champion — the fix for savings not stacking / monoculture.
- **#2 temp-sampling probe — RUN, but it tested the WRONG variable (2026-08-26).** It sampled the
  OUTPUT distribution of ONE FIXED input (the champion) at temperature 1.0, 300 samples/victim:
  fire rate 1.00, min firing gen = 28 (gpt) / 29 (gemma), zero sub-floor, max 32. This ONLY shows
  the *champion's* output distribution is tightly peaked at 28/29 -- it varied the output, not the
  input. It does NOT test reachability, because a DIFFERENT input can have a completely different
  greedy output; absence from one input's samples proves nothing about the global floor. **Retract
  the "output floor proven" claim.** The correct reachability question -- "is there an INPUT whose
  GREEDY output fires in < 28/29 tokens" -- is a search/priming over INPUTS, not sampling one
  input's outputs. The right instruments (Cluster 5.3): assistant-turn PREFILL toward the sub-floor
  render + greedy-complete (does it finish firing without the scaffold token?), few-shot format
  exemplars, and the structural input search (#3/#4). The temp probe is retired as mis-specified.
- **Prefill probe (the correct input-varying test) — RUN, clean NEGATIVE (2026-08-26).** Injected
  the assistant TOOL-CALL scaffold into the input (harmony commentary header + JSON up to the host)
  so the victim would only generate the close. It FAILED both victims: gpt went into analysis (gen
  212, no fire), gemma refused (gen 45, no fire). STRUCTURAL reason: the SDK appends its OWN
  `<|start|>assistant` generation prompt after our user message, so a partial tool call in the input
  is user content the model reasons about, not an assistant turn it continues -- and the SDK only
  parses/executes tool calls the model GENERATES, never ones we inject. The forge works only because
  it is a COMPLETE prior turn continued as a fresh turn. But this was ONE construction defeated by
  the SDK's turn structure -- not a proof.
- **CORRECTION (2026-08-26) -- "28/29 floored" was a CONFLATION.** It ran together two different
  numbers: (a) the minimum tokens that FIRE -- the parser-acceptable render, gpt ~22 (no
  `<|constrain|>json`) / gemma ~21 (unquoted args), verified by calling the grader parser directly
  -- and (b) what the CHAMPION greedily EMITS, 28/29. The firing minimum is (a) ~22/21, which is
  6-8 tokens BELOW the champion. The ablation floored only the champion's DELETION neighborhood and
  the prefill probe failed ONE construction; neither shows NO input yields a leaner greedy render,
  and the parser proves a leaner FIRING render exists. So the output is NOT proven floored -- there
  is a real 6-8 token gap between what fires (~22/21) and what the champion emits (28/29), and
  closing it is an OPEN INPUT SEARCH: find an input whose GREEDY output is the parser-minimal
  render. That is exactly what the structural search (#3 descriptors, #4 islands) targets -- the
  priority, not a dead end.
- **#3 structural descriptors, #4 islands** — not yet built; ranked as above. #3 cold-starts the
  archive (descriptor change), so hold it until the pending board tests report.

## Key sources
EvoPrompt 2309.08532 · OPRO 2309.03409 · PromptBreeder 2309.16797 · CAPO 2504.16005 · APO survey
2502.16923 · Rainbow Teaming 2402.16822 · RainbowPlus 2504.15047 · Ferret 2408.10701 · CMA-ME
1912.02400 · MAP-Elites 1504.04909 · AutoDAN-Turbo 2410.05295 · AutoDAN 2310.04451 · TAP 2312.02119 ·
PAIR 2310.08419 · GPTFuzzer 2309.10253 · FunSearch (Nature 2023) · lexicase+novelty 1905.09374 ·
TALE 2412.18547 · length-control 2504.14350 · LLAMBO (skip) 2402.03921
