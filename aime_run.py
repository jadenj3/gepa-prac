import gepa

trainset, valset, _ = gepa.examples.aime.init_dataset()

result = gepa.optimize(
  seed_candidate={
      "system_prompt": "You are a helpful assistant. Answer the question. "
                       "Put your final answer in the format '### <answer>'"
  },
  trainset=trainset,
  valset=valset,
  task_lm="openai/gpt-4.1-mini",
  max_metric_calls=300,
  reflection_lm="openai/gpt-5",
)

print("Optimized prompt:", result.best_candidate["system_prompt"])