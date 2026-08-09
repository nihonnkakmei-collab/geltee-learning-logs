# Geltee infinite learning loop

The loop starts from the supplied checkpoint `geltee_v227_v171_small_vector_search_model.pt`. Its local path is intentionally not recorded in this public repository.

SHA-256: `9102BE9ED756F713F6DF83B200F97A0F22A53ACEE73465610E8EF91A4D2538FE`

Each step samples broad training data and a separate holdout set. The fixed 100-case Geltee gate is evaluation-only and is never used as training data. A candidate is promoted only if it does not reduce the gate score and improves holdout answer loss by at least 0.3%. Before every promotion, the existing champion is saved locally for rollback. The latest status and log are overwritten in `logs/`. Every 10 steps, the runner commits and pushes those files. Checkpoints remain local because the source model exceeds GitHub's normal file-size limit.

`GPT-1を超えた` can only be claimed after a common benchmark against a reproduced GPT-1 baseline. The internal 100-case Geltee gate is tracked separately and is not presented as proof of that claim.

Stop the loop by creating `STOP` in this repository or by terminating the process.
