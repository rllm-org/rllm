"""Is greedy decoding on this vLLM deterministic, and does batch composition break it?

Prompt-level test, no agent loop: the same prompt is sent
  (a) serially, one request at a time      -> batch size 1 every time
  (b) inside a burst of 32 concurrent      -> batch composition varies
and the outputs are compared byte for byte.

If (a) is self-identical and (b) is not, batch composition is the noise source
and nothing short of serial decoding removes it.
"""
import json, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
import openai

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8160/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REPS = 8
FILLER = 31  # concurrent decoys surrounding the probe

PROMPT = ("You are an expert financial analyst. Explain, in careful step-by-step detail, "
          "how to compute the compound annual growth rate of revenue from 2015 to 2020, "
          "why it differs from the simple average annual growth rate, and what a reader "
          "should watch for when a company restates prior-period figures.")

c = openai.OpenAI(base_url=BASE, api_key="yyy")

def gen(p, maxtok=400):
    r = c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": p}],
                                  temperature=0, top_p=1.0, max_tokens=maxtok, seed=1234)
    return r.choices[0].message.content or ""

def h(s): return hashlib.md5(s.encode()).hexdigest()[:10]

print(f"endpoint: {BASE}\nreps: {REPS}  concurrent filler: {FILLER}\n")

serial = [gen(PROMPT) for _ in range(REPS)]
print("=== (a) serial, batch size 1 ===")
hs = [h(x) for x in serial]
print(f"  hashes: {hs}")
print(f"  distinct outputs: {len(set(serial))}/{REPS}  -> " +
      ("DETERMINISTIC" if len(set(serial)) == 1 else "NONDETERMINISTIC"))

def burst(_):
    with ThreadPoolExecutor(max_workers=FILLER + 1) as ex:
        futs = [ex.submit(gen, PROMPT)]
        for i in range(FILLER):
            futs.append(ex.submit(gen, PROMPT + f" (variant {i}) " + "context " * (i % 17), 200))
        return futs[0].result()

batched = [burst(i) for i in range(REPS)]
print("\n=== (b) same prompt inside a 32-wide concurrent burst ===")
hb = [h(x) for x in batched]
print(f"  hashes: {hb}")
print(f"  distinct outputs: {len(set(batched))}/{REPS}  -> " +
      ("DETERMINISTIC" if len(set(batched)) == 1 else "NONDETERMINISTIC"))

print("\n=== cross-check ===")
print(f"  serial output == batched output : {serial[0] == batched[0]}")
allout = set(serial) | set(batched)
print(f"  distinct outputs overall        : {len(allout)}/{2*REPS}")
if len(set(serial)) == 1 and len(set(batched)) > 1:
    print("\n  VERDICT: batch composition is the noise source.")
elif len(set(serial)) > 1:
    print("\n  VERDICT: nondeterministic even at batch size 1 - the cause is not batching.")
else:
    print("\n  VERDICT: deterministic in both regimes - the eval noise comes from elsewhere.")
