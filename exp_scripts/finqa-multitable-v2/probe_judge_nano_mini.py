"""Paired, fixed-answer judge probe; never reruns or changes policy rollouts."""
import concurrent.futures
import json
import os
from pathlib import Path
import random
import statistics
import sys

import finqa_eval as evaluator


def score(model, samples, output):
    evaluator.MULTI_TABLE_JUDGE_MODEL = model
    evaluator.JUDGE_FINISH_LOG = str(output / (model + '_finish.tsv'))
    rows = []
    for sample in samples:
        task = sample['episode']['task']
        answer = sample['episode']['artifacts']['answer']
        prompt = f"question : {task.get('core_question') or task['question']}\nmodel response : {answer}\nlabel : {task['ground_truth']}"
        value, rubric = evaluator._call_judge(evaluator.MULTI_TABLE_CORRECTNESS_PROMPT, prompt, multi_table=True)
        row = {'split': sample['split'], 'idx': sample['idx'], 'historical_nano': sample['historical_nano'], 'score': value, 'rubric': rubric}
        rows.append(row)
        with (output / (model + '.jsonl')).open('a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(model, sample['split'], sample['idx'], value, flush=True)
    return rows


if __name__ == '__main__':
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260915)
    samples = []
    for split in ['multi_val', 'multi_test']:
        result = json.loads((root / (split + '.json')).read_text())
        files = sorted((root / ('episodes_' + split) / 'episodes').glob('*.json'))
        for file in rng.sample(files, 12):
            episode = json.loads(file.read_text())
            idx = episode['eval_idx']
            historical = next(i['signals']['accuracy'] for i in result['items'] if i['idx'] == idx)
            samples.append({'split': split, 'idx': idx, 'historical_nano': historical, 'episode': episode})
    (output / 'samples.json').write_text(json.dumps(samples, indent=2))
    # Separate processes isolate evaluator's model globals; at most two judge calls at once.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(score, model, samples, output) for model in ['gpt-5.4-nano', 'gpt-5.4-mini']]
        nano, mini = [f.result() for f in futures]
    valid = [(a, b) for a, b in zip(nano, mini) if a['rubric'] and b['rubric']]
    delta = [b['score'] - a['score'] for a, b in valid]
    summary = {'sample_count': len(samples), 'valid_pairs': len(valid),
               'nano_mean': statistics.mean(a['score'] for a, b in valid),
               'mini_mean': statistics.mean(b['score'] for a, b in valid),
               'mini_minus_nano_mean': statistics.mean(delta),
               'mean_absolute_difference': statistics.mean(abs(d) for d in delta),
               'difference_ge_0.1': sum(abs(d) >= .1 for d in delta),
               'threshold_0.9_disagreements': sum((a['score'] >= .9) != (b['score'] >= .9) for a, b in valid),
               'nano_repeat_mean_absolute_difference': statistics.mean(abs(a['score']-a['historical_nano']) for a, b in valid)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
