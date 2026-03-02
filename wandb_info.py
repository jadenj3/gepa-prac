import wandb

api = wandb.Api()
runs = api.runs("jaden8000-123/gepa_aime")

scores = []
target_iteration = 16  # change this

def get_valset_score(target_iteration):
    for run in runs:
        history = run.scan_history(keys=["iteration", "best_valset_agg_score"])
        for row in history:
            if row.get("iteration") == target_iteration and row.get("best_valset_agg_score") is not None:
                scores.append(row["best_valset_agg_score"])
                break

    print(f"Iteration {target_iteration}: {len(scores)} runs, avg = {sum(scores) / len(scores):.4f}")

get_valset_score(target_iteration)