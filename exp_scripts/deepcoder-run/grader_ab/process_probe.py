"""Measure forkserver/Manager lifecycle without executing benchmark code."""
import concurrent.futures
import json
import os
import statistics
import time

from compatible_grader import install_grader_context


def child(result):
    result.append(1)


def one(_):
    import rllm.rewards.code_reward as cr
    marks = [time.perf_counter()]
    manager = cr.Manager(); marks.append(time.perf_counter())
    result = manager.list(); marks.append(time.perf_counter())
    process = cr.multiprocessing.Process(target=child, args=(result,)); marks.append(time.perf_counter())
    process.start(); marks.append(time.perf_counter())
    process.join(20); marks.append(time.perf_counter())
    alive = process.is_alive()
    if alive: process.kill(); process.join()
    ok = list(result) == [1]
    manager.shutdown(); marks.append(time.perf_counter())
    return {"manager": marks[1]-marks[0], "list": marks[2]-marks[1],
            "construct": marks[3]-marks[2], "start": marks[4]-marks[3],
            "join": marks[5]-marks[4], "shutdown": marks[6]-marks[5],
            "total": marks[6]-marks[0], "alive": alive, "ok": ok}


if __name__ == "__main__":
    install_grader_context()
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        rows=list(pool.map(one, range(48)))
    out={}
    for key in ("manager","list","construct","start","join","shutdown","total"):
        vals=sorted(row[key] for row in rows)
        out[key]={"p50":statistics.median(vals),"p90":vals[int(.9*len(vals))],"max":max(vals)}
    out.update(wall=time.perf_counter()-start, failures=sum(not r["ok"] or r["alive"] for r in rows))
    print(json.dumps(out, indent=2))
