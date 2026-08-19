---
name: rllm-simplify
description: 'Pre-PR diff diet for rLLM: audit an outgoing branch/PR (or stacked PR) for the bloat patterns this repo measurably accumulates — test accretion, implementation-encoding tests, scratchpad comments/docstrings, surface accretion, dual paths — prove every candidate with call-site evidence, apply behavior-preserving cuts, and leave a journal. Invoke after implementing a feature and before submitting the PR, or on an existing PR by number/branch.'
---

# /rllm-simplify — shrink the PR without changing what it does

Turn "this diff feels heavy" into evidence-backed deletions. Measured drift on this repo's nightly branch (99 commits over one month): 4.8 added lines per deleted line, tests the single largest sink (~36% of added Python), a quarter of added non-blank Python lines were comments/docstrings, and only 5 of 99 commits were net-negative. This skill is the counterweight: it runs on one outgoing PR and removes weight that the feature does not need, before review.

This is guidance, not a checklist to recite. Follow the code, prefer a few proven cuts over many thin ones, and keep behavior identical. You are editing a diff you (or another agent) just wrote — sunk cost is not evidence; "I already tested it" is not a reason to keep a test.

## Scope the diff first

- Resolve the true base. For a stacked PR, diff against the merge-base with the PR's **own base branch** (`gh pr view N --json baseRefName`), never against the integration branch — you audit only this PR's commits.
- Enumerate consumers of every new public symbol **including later PRs in the stack**: check out or `git grep` the stack tip. A later PR is a production consumer; cutting API it uses breaks the stack. Conversely, a symbol only the stack tip's *tests* use is still test-only. At the stack TIP the question inverts: nothing downstream can appear later, so a repo-wide grep is definitive — run the speculative-surface sweep there (a new public symbol with no production caller is *provable* at the tip, only suspectable below it).
- In a stack, read every parent PR's simplification journal before rebasing: the journals forecast your conflicts and their resolution rules, and they carry handoff items addressed to your layer. Report each conflict's true origin (trial-rebase against pre- and post-diet parent heads when unsure).
- Inventory the diff before judging it: split added lines into production / tests / CI+config / docs, compute the test:production added-line ratio, and count the new test functions. Above ~1.5:1 the burden of proof flips — every test file must justify itself (the ratio alone is never the finding; the unjustified files are). Lines flag; per-test evidence — kill sets, superseded-test answers — decides which tests go.
- **Baseline first**: run the targeted suites once before touching anything. A red baseline is a stop-and-report, not something to fix in a simplification pass. Distinguish red from an environment gap: collection errors or failures in files the PR never touched usually mean a missing extra/plugin (e.g. `uv run --with pytest-asyncio …`) — fix the invocation, note it in the journal, and treat the corrected run as the baseline. When the gap sits inside the PR's own diff (e.g. a broken import-guard in a new fixture), characterize it from outside — a throwaway pytest plugin via `-p`, never an edit to the file under audit before its baseline exists.

## Ground rules this skill enforces

Testing discipline:

- **Red evidence for every surviving claim**: a test that cannot be demonstrated to fail on pre-change (or mutated) code is presumed written-to-pass — a finding, not coverage.
- **Placement before authorship**: a case that varies only inputs/config of already-tested behavior extends that test (`@pytest.mark.parametrize` or an added assertion), not a new function; new functions are for genuinely new behavior.
- **Mock only boundaries you don't own** (network, provider SDKs, GPUs, subprocess launch, clock). A stub that re-implements the logic under test makes the test tautological.

Comment discipline (violations are review findings):

- A comment states a constraint or intent the code can't show. One brief intent comment per logical block — never per-line micro-comments, never restating the next line.
- **Never fix-narration**: comments shaped like "X didn't work, so we now Y" narrate the edit; that text belongs in the commit message, not the code.

Structure facts that decide where code belongs:

- Fireworks subclasses Tinker everywhere — duplicated tinker logic in fireworks files is a defect, not a convenience; check whether a tinker-layer edit is inherited by the fireworks subclass.
- verl parity is deliberate work: a tinker-only feature ships with a tested gate, not a silent gap.
- Dataset-specific bridges live in `cookbooks/`, never core `rllm/`.
- Dead zones (`rllm/sdk/`, `rllm/trainer/deprecated/`, `rllm/agents/` shim) — new code building on them is a finding by default.
- `rllm-model-gateway/` is a separate uv package with its own test tree.
- CI runs pre-commit plus a handful of path-gated smoke files only, so "CI passes" proves nothing about a deleted test's safety — local targeted pytest does.

## The bloat ledger — patterns this repo actually produces

### 1. Test accretion (the largest measured sink)

- **The superseded-test question, asked per new test**: which existing test does this replace, extend, or overlap? A PR that only ever appends test functions is the pattern this skill exists for. Search the existing tree (`git grep -l <symbol> tests/ rllm-model-gateway/tests/`) before accepting any new test file as necessary.
- N near-identical functions sharing a setup skeleton = one parametrized test. Choose params so each isolates ONE guard/branch — a fold done this way frequently *adds* coverage: a compound test that violates two guards at once can prove neither individually, while the parametrized split can. Fixture builders copy-pasted across new test files belong in the nearest `conftest.py` — or are a sign the two files should be one.
- **Framework-restating tests**: asserting that a Pydantic model rejects a missing required field, that `Literal["compact"]` rejects `"x"`, that a dataclass round-trips through its own `model_dump` — these pin pydantic/stdlib behavior, not ours. Delete unless the field's presence *is* the wire contract a peer implementation depends on (then one test for the contract, not one per field).
- **Mirror tests**: the same fact proven at two layers in one PR (e.g. store-level and client-level tests that both assert the same serialization round-trip) — or proven again against a PARENT PR's test files, the routine shape when a stack wraps a previously-added element in a container (the child re-tests the element through the container). Run the mirror check against the parent PRs' tests, not just this diff. Keep the layer where a regression would actually originate; delete the echo.
- **CI reachability is evidence, cheap and decisive**: for every new test file or directory, check whether any workflow can ever run it (`grep -rF <path-fragment> .github/workflows/`). CI is path-gated to a handful of files, so a new test *directory* is almost always dead in CI — which both weakens "we need it for regressions" and strengthens folding survivors into an already-gated file (the fold can turn zero CI coverage into real CI coverage). Before folding a whole directory away, check the stack tip: a later PR may add files to that directory — the stack-consumer rule that protects symbols protects test homes too.
- **Environment-gated tests that silently skip**: a test discovering fixtures from `~/.rllm/...` or an env var skips everywhere but the author's machine and counts as zero coverage. Three sanctioned outcomes: commit a small sanitized fixture so it runs everywhere; move it out of the default suite into an explicitly documented release-sweep marker; or — when it was a one-shot verification harness whose numbers are already recorded in the PR body — simply delete it, the evidence outlives the script. Never leave it looking like CI coverage.
- **The deletion question** for any test you keep: *if the behavior it pins broke, what would a user or downstream PR observe?* No concrete answer → it is not load-bearing; fold it or cut it. Fixture lines discharge the burden differently from test functions: a fat fixture is justified by the mutants only it kills, never by how many tests share it.

### 2. Implementation-encoding tests

- Byte-exact comparisons of internal representations where the contract is semantic equality; assertions on private attributes, mock call sequences, internal ordering that no consumer observes.
- Stubs that re-implement the logic under test (a fake store that dedups, testing the deduper) — tautological; assert through the real path or scope the claim down.
- A test that had to be rewritten in the same PR that only refactored internals is confessing: it encoded the old implementation. Its replacement probably encodes the new one.

### 3. Comment and docstring excess

- Scratchpad narration ("we tried X…", "note that this preserves…"), fix-narration, step-numbered comments, and PR-body essays (design rationale, complexity analysis, verification claims) living in code. Rationale belongs in the PR body/commit message; code keeps only constraints the code can't show.
- Docstrings restating the signature or the function name; module docstrings that duplicate the class docstring below them; comment altitude violations (per-line commentary where one block-intent line belongs).
- The measured base rate is ~25% of added Python lines; a well-groomed diff sits well under that. Trim toward intent-only.

### 4. Surface accretion

- A new helper/module/class where an existing structure could absorb the change — the question is always "what existing thing should have grown or shrunk instead?"
- Dual legacy/new code paths where the legacy path has no remaining production consumer, or where the split exists only so old tests keep passing unmodified (that's test-driven bloat, doubly cut).
- Single-use `_private` helpers that read worse than inlining; re-export/compat layers; config knobs with exactly one value ever passed; defensive fallbacks for states that cannot occur (`hasattr` on our own types, `or []` on fields with defaults); speculative generality (registries with one entrant, ABCs with one implementation) — version/format discriminators on **wire formats** are the legitimate exception.
- Plumbing a kwarg through 4 layers to a single consumer — consider whether the consumer can read it from where it already lives.

### 5. Where things belong (structural findings, usually handoffs not edits)

Shared-kernel files (`rllm/types.py`, `trainer/unified_trainer.py`, `gateway/manager.py`, …) growing per-feature branches; tinker-only behavior without a verl gate; coupled-subsystem edits without smoke-testing the other side. Flag these in the journal as findings for the owning reviewers, with evidence — restructuring across domains is beyond a simplification pass unless the fix is local and behavior-preserving.

## Prove, then cut

For every candidate: `rg`/`git grep` the exact symbol, config key, wire string across production (`rllm/`, `rllm-model-gateway/src/`, `examples/`, `cookbooks/`, and later stack PRs) vs non-production (tests, docs, comments) corpora, then read the call sites.

For the PR's new *tests*, make the evidence quantitative up front with a **mutation kill matrix**: write a dozen-plus hand mutations of the PR's own production lines (flip a guard, drop a validator, off-by-one a slice, swap an inherited field for a recomputed one) and record which tests kill which mutant. Zero-kill tests are cut candidates by evidence, not taste; tests with identical kill sets are mirror/fold candidates; a mutant NO test kills is a coverage hole the consolidation should close. For add-only PRs this matrix is the only real evidence available (revert-to-base reds everything vacuously), so it is the opening move, not a closing check.

The matrix cuts both ways, and its vetoes are worth as much as its licenses: expect it to protect cuts that look attractive by eye (a fat fixture often earns every line by the field-preservation mutants only it kills) and to license cuts that read as core contract (a test whose kill set is a proper subset of another's is redundant no matter how important it looks). Mutant-writing takes rounds, not one act — and before cutting any zero-unique-kill test, run a **probe round**: try to write one more mutant that only that test would catch. If you cannot invent one, the cut is safe; if you can, you just saved real coverage.

Three refinements the matrix needs on real PRs: (i) **parity suites blunt it** — a test asserting equivalence between two forms (flat vs delta, legacy vs new) is blind to any mutant that shifts both forms together, so classify each surviving mutant as form-specific (a real hole) or form-agnostic (invisible by construction) before reporting; a probe for a parity test must be one-sided, and when no one-sided probe can exist, that impossibility is itself the licence to cut. (ii) **A survivor is not automatically a hole**: when the mutated production branch provably does nothing (both paths byte-identical with and without it), the survivor is an *equivalent mutant* — a production dead-code finding, often the most valuable output of the whole pass. (iii) **After any production-side cut, re-express the mutants anchored on the changed lines** before the re-red run, or the harness reports fake regressions and silent pattern misses.

Reject or downgrade when:

- a production caller exists (removal would be a feature decision, not a cleanup);
- the test is the **only** red-proven coverage of a behavior a user relies on;
- the "simplification" changes observable behavior, relaxes validation a wire format or store depends on, or renames public API;
- the idea is right but tiny relative to churn — note it in the journal instead of thrashing the diff.

When uncertain, keep it and record why in the journal's rejected-candidates section. A kept candidate with a written reason is a good outcome; a silent keep is not.

## Apply

- Behavior-preserving only. The PR must do exactly what it did before, minus weight.
- **Re-red consolidated tests** — two mechanics, chosen by whether the pinned behavior predates the PR. (1) Behavior the PR *modified*: revert the implementation files to the PR's base (`git restore --source=<merge-base> --worktree -- <impl files>`), confirm the surviving suite goes red on what it claims to pin, then `git restore --source=HEAD` — and commit or stash the diet FIRST, because that restore overwrites uncommitted edits in the same files. (2) Symbols the PR *added*: revert-to-base is vacuous (everything dies at import); **mutate** the specific pinned behavior instead and confirm the survivor reds, then undo — impl and test files are disjoint here, so there is no commit-first hazard. If you built the kill matrix during discovery, re-running it after the diet IS the re-red: no mutant previously killed may survive. Script the harness to restore the implementation file in a `finally` — an interrupted run must not leave the tree mutated. Paste the red evidence into the journal. A consolidation that stays green under the revert/mutation deleted the coverage, not the duplication.
- Run the targeted suites for everything touched (with the right extras) plus `pre-commit run --files <changed>`. Gateway package tests run from the repo root: `uv run --with pytest-asyncio pytest rllm-model-gateway/tests/unit/` (the plugin is not in the root venv). Never mix `tests/` and the gateway package in one pytest process — `tests/conftest.py`'s `sys.modules` stubs leak.
- Commit the simplification as its own commit(s) on the PR branch — `refactor(scope): …` / `test(scope): …` — never squashed into the feature commits, so reviewers can see the diet separately. Do not push unless asked to.
- Touch only files in the PR's diff, plus deletions elsewhere that the PR's changes make dead (state the evidence).

## Journal

Write `tmp/simplify/<branch>.md` (gitignored scratch — keep it, don't commit it):

- one entry per applied cut: **what** (files, −lines), **why** (which ledger pattern), **evidence** (call-site proof, superseded test named), **risk** and how it was discharged (re-red run, suite results);
- a **rejected candidates** section: considered, kept, and the reason;
- before/after diffstat of the PR (base..head), production and test split;
- exact test commands run with pass/fail counts;
- a boundary check: list edited files by owning subsystem; anything outside the domain that drove the PR is flagged for its owners' review, with handoff items for work you found but did not do;
- stacked-children impact: trial-rebase each child onto BOTH the pre-diet and post-diet heads — only conflicts unique to the post-diet head are attributable to this pass; report the rest with their true origin.

The journal's summary becomes the simplification commit's body and feeds the PR description update.

## Done means

- The PR's diff is strictly lighter (or every keep is justified in the journal); behavior unchanged.
- All targeted suites green at the new head; consolidated tests re-proven red against base or mutation; pre-commit clean.
- Journal written; stacked children (if any) still apply cleanly or their rebase need is reported.
