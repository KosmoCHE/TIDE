# Development Notes


## 数据格式

```python

result: Dict[str, Any] = {
    "query": str,
    "all_accuracies": List[int],
    "avg_accuracy": float,
    "pass_at_k": float,
    "traj_rollouts": [
        {
            "rollout_idx": int,
            "rollout_results": rollout_trajectory
        },
        # ... more rollouts
    ]
}

rollout_trajectory: Dict[str, Any] = {
    "query": str ,
    "success": bool ,
    "loop": [
        {
            "step": int,
            "action": str,
        },
        ...
    ],
    "steps": [
        {
            "last_step_feedback": {
                "is_valid": bool,
                "message": str
            },
            "model_input": str,
            "response": str,
            "observation": str,
            "input_state": str,
            "true_state": str,
            "analysis": str,
            "action": str,
            "tokens": List[str],
            "token_ids": List[int],
            "action_space_entropy": float,

            "token_entropy_stats": {
                "analysis_stats": {
                    "mean": float,
                    "std": float,
                    "max": float,
                    "min": float,
                    "raw": List[float]
                },
                "action_stats": dict[str, float]
                },
            "env_feedback":{
                "action_is_valid": bool,
                "accuracy": int,
                "level": str,
                "done": bool
            }
        },
        ...
    ]
}

```