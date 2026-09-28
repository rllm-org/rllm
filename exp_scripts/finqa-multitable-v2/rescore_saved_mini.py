"""Rejudge saved answers only, using the unchanged upstream multi-table rubric."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import finqa_eval as judge


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(item):
    path, historical = item
    episode = json.loads(path.read_text())
    task = episode['task']
    answer = episode['artifacts']['answer']
    if not answer or not task.get('ground_truth'):
        raise ValueError(f'Missing scoring input: {path}')
    prompt = f"question : {task.get('core_question') or task['question']}\nmodel response : {answer}\nlabel : {task['ground_truth']}"
    score, rubric = judge._call_judge(judge.MULTI_TABLE_CORRECTNESS_PROMPT, prompt, multi_table=True)
    if not all(isinstance(rubric.get(k), (int, float)) and math.isfinite(rubric[k]) and 0 <= rubric[k] <= 100 for k in judge.CORRECTNESS_WEIGHTS):
        raise ValueError(f'Invalid judge verdict, not a policy failure: {path}')
    return {'idx': episode['eval_idx'], 'score': score, 'is_correct': score >= .9,
            'historical_nano': historical, 'rubric': rubric,
            'source_episode': str(path.resolve()), 'source_sha256': digest(path)}


if __name__ == '__main__':
    source, output = map(Path, sys.argv[1:3])
    requested_tags = set(sys.argv[3:])
    output.mkdir(parents=True, exist_ok=True)
    judge.MULTI_TABLE_JUDGE_MODEL = 'gpt-5.4-mini'
    all_tags = [('Base', 'base')] + [(str(i), f'global_step_{15*i}') for i in range(1, 6)]
    for epoch, tag in all_tags:
        if requested_tags and tag not in requested_tags:
            continue
        if not (source / tag / 'COMPLETE').exists():
            raise ValueError(f'Source evaluation incomplete: {tag}')
        dest = output / tag
        dest.mkdir(exist_ok=True)
        for split, expected in [('multi_val', 126), ('multi_test', 131)]:
            original = source / tag / (split + '.json')
            items = json.loads(original.read_text())['items']
            historical = {i['idx']: i['signals']['accuracy'] for i in items}
            paths = sorted((source / tag / ('episodes_' + split) / 'episodes').glob('*.json'))
            indices = [json.loads(p.read_text())['eval_idx'] for p in paths]
            if len(paths) != expected or set(indices) != set(historical) or len(set(indices)) != expected:
                raise ValueError(f'Incomplete or duplicate source indices: {tag}/{split}')
            journal = dest / (split + '.jsonl')
            rows = [json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
            if len({r['idx'] for r in rows}) != len(rows):
                raise ValueError('Duplicate rescoring rows')
            by_idx = {json.loads(p.read_text())['eval_idx']: p for p in paths}
            for row in rows:
                if row['source_sha256'] != digest(by_idx[row['idx']]):
                    raise ValueError('Source changed since previous scoring')
            done = {r['idx'] for r in rows}
            judge.JUDGE_FINISH_LOG = str(dest / ('judge_finish_' + split + '.tsv'))
            manifest = {'operation': 'fixed-answer judge-only rescoring', 'judge': judge.MULTI_TABLE_JUDGE_MODEL,
                        'epoch': epoch, 'split': split, 'expected': expected,
                        'source_result_sha256': digest(original),
                        'rubric_sha256': hashlib.sha256(judge.MULTI_TABLE_CORRECTNESS_PROMPT.encode()).hexdigest(),
                        'max_output_tokens': 5000, 'reasoning_effort': 'medium',
                        'concurrency': 8, 'success_threshold': .9}
            (dest / (split + '.manifest.json')).write_text(json.dumps(manifest, indent=2))
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(evaluate, (p, historical[idx])) for idx, p in by_idx.items() if idx not in done]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    with journal.open('a') as fh:
                        fh.write(json.dumps(row) + '\n')
                    rows.append(row)
                    print(tag, split, f'{len(rows)}/{expected}', flush=True)
            correct = sum(r['is_correct'] for r in rows)
            result = {'epoch': epoch, 'total': len(rows), 'correct': correct,
                      'success_rate': correct / expected,
                      'mean_reward': sum(r['score'] for r in rows) / expected,
                      'items': sorted(rows, key=lambda r: r['idx'])}
            (dest / (split + '.json')).write_text(json.dumps(result, indent=2))
        (dest / 'COMPLETE').touch()
    print('ALL_MINI_RESCORE_COMPLETE', flush=True)
