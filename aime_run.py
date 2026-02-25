import time

import gepa


class TimingCallback:
    def __init__(self):
        self._iter_start = 0.0
        self._eval_start = 0.0
        self._proposal_start = 0.0
        self._valset_start = 0.0

    def on_iteration_start(self, event):
        self._iter_start = time.time()
        print(f"\n--- Iteration {event['iteration']} ---")

    def on_evaluation_start(self, event):
        self._eval_start = time.time()

    def on_evaluation_end(self, event):
        elapsed = time.time() - self._eval_start
        label = "subsample (traces)" if event["has_trajectories"] else "subsample (new candidate)"
        print(f"  {label} [{event['candidate_idx'] or 'new'}] ({len(event['scores'])} examples): {elapsed:.1f}s")

    def on_proposal_start(self, event):
        self._proposal_start = time.time()

    def on_proposal_end(self, event):
        elapsed = time.time() - self._proposal_start
        print(f"  Reflection LM: {elapsed:.1f}s")

    def on_valset_evaluated(self, event):
        elapsed = time.time() - self._eval_start
        print(f"  Valset eval ({event['num_examples_evaluated']} examples): {elapsed:.1f}s  score={event['average_score']:.3f}")

    def on_iteration_end(self, event):
        elapsed = time.time() - self._iter_start
        status = "accepted" if event["proposal_accepted"] else "rejected"
        print(f"  Total iteration ({status}): {elapsed:.1f}s")


trainset, valset, _ = gepa.examples.aime.init_dataset()

result = gepa.optimize(
    seed_candidate={
        "system_prompt": "You are a helpful assistant. Answer the question. "
        "Put your final answer in the format '### <answer>'"
    },
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    max_metric_calls=5000,
    reflection_lm="openai/gpt-5",
    raise_on_exception=False,
    callbacks=[TimingCallback()],
)

print("Optimized prompt:", result.best_candidate["system_prompt"])
