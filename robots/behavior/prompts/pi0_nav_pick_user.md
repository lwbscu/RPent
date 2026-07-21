Call `pi0_nav_pick` exactly once with instruction={{ pi0_instruction }}. Do not
provide hand, threshold, required frames, or max_chunks, and do not call another
action primitive before it returns. If it returns a formal paused held-object
handoff, continue with `inspect_post_pick_state` and the bounded pre-press tools.
Do not repeat navigation or grasping, and do not press. End only after
`state_checkpoint_2` is saved or a fail-closed guard blocks the phase.
