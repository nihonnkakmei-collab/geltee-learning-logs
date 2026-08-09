# Geltee infinite learning loop

The loop starts from the exact checkpoint supplied by the user:

`C:\Users\matsu\Downloads\battle_v227_v171_small_vector_search_candidate\geltee_v227_v171_small_vector_search_model.pt`

SHA-256: `9102BE9ED756F713F6DF83B200F97A0F22A53ACEE73465610E8EF91A4D2538FE`

Each step creates a candidate from the current champion, evaluates the fixed 100-case Geltee gate, and promotes only a non-regressing candidate. The latest status and log are overwritten in `logs/`. Every 10 steps, the runner commits and pushes those files. Checkpoints remain local because the source model exceeds GitHub's normal file-size limit.

`GPT-1を超えた` can only be claimed after a common benchmark against a reproduced GPT-1 baseline. The internal 100-case Geltee gate is tracked separately and is not presented as proof of that claim.

Stop the loop by creating `STOP` in this repository or by terminating the process.

