Solve the full task with VLA navigation and visual planner tools. Before
grasping, call `pi0_navigate_to` in repeated short segments with
instruction={{ pi0_instruction }} unchanged and max_chunks=4. After every
segment, review all three returned views and observe again before deciding
whether to call another segment. Navigation may execute whole-body posture
commands but keeps both grippers latched;
never claim or attempt a grasp with pi0_navigate_to.

For the grasp phase call `pi0_pick` with hand={{ pi0_hand }},
instruction={{ pi0_instruction }}, max_chunks={{ pi0_max_chunks }},
stop_on_closure_candidate=true, and post_candidate_chunks=4. `max_chunks` is the
total cap, including the four post-candidate chunks. Visually review the
returned head and wrist images before continuing. Before that call, require a
fresh image with the radio clearly visible plus a successful plan-only and
executed selected-hand pre-grasp move 12--20 cm from it; otherwise reposition
the base with another bounded `pi0_navigate_to` segment and re-observe. Do not
use planner coordinate navigation. Never call `pi0_pick` as a substitute for
post-grasp button localization.
