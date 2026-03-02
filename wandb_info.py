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


def get_latest_valset_scores():
    latest_scores = []
    for run in runs:
        score = run.summary.get("best_valset_agg_score")
        if score is not None:
            latest_scores.append(score)
            print(f"  {run.name}: {score:.4f}")
    if latest_scores:
        print(f"\n{len(latest_scores)} runs, avg = {sum(latest_scores)/len(latest_scores):.4f}")
    else:
        print("No runs with best_valset_agg_score found")


get_latest_valset_scores()