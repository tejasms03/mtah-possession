# Features

All features come from the **tactical ATD** (plus Smart Tagging for a few game-level flags).
Nothing is extracted from the video. The same code computes them for our match and for degraded
IDSSE, so the model sees identical inputs from both.

Raw ATD per frame: ball `x, y` (NaN on 41% of graded frames) and, per track slot, `x, y, speed`
and team. Slots have no identity and swap, so no feature refers to a slot; every player feature
is "the team's nearest player", "count of players", or an aggregate over a team.

The model gets these for every frame in a window of ±2 s (101 frames) around the frame it labels.

## 1. Ball (from the IMM filter, stage 3)

| feature | meaning |
|---|---|
| `ball_x`, `ball_y` | smoothed position, metres |
| `ball_vx`, `ball_vy`, `ball_speed` | smoothed velocity |
| `ball_accel` | change in velocity (spikes = touches) |
| `p_carried`, `p_rolling`, `p_airborne` | which motion model explains the ball best |
| `ball_observed` | 1 if the ATD has the ball this frame, 0 if filled by prediction |
| `ball_uncertainty` | filter uncertainty (grows during gaps) |
| `gap_len_s` | how long the ball has been missing, 0 if observed |
| `dist_touchline`, `dist_goalline` | to the nearest line, negative when outside |
| `out_of_bounds` | ball outside the pitch |

## 2. Ball vs each team (computed twice: `_H` and `_A`)

| feature | meaning |
|---|---|
| `d_nearest` | distance from ball to the team's nearest player |
| `dv_nearest` | velocity mismatch between ball and that player |
| `coupling` | `exp(-d²/2σd²)·exp(-|Δv|²/2σv²)·(1 − p_airborne)` |
| `n_within_2m`, `n_within_5m` | team players near the ball |
| `touch` | touch impulse this frame with this team nearest |
| `coupled_run_s` | how long this team has been continuously coupled |
| `last_touch_ago_s` | time since this team's last touch |

Plus `contested` (both teams pass the gates) and `d_nearest_H − d_nearest_A`.

## 3. Team shape (per team, works without the ball)

| feature | meaning |
|---|---|
| `centroid_x`, `centroid_y` | mean position, x flipped so + is always attacking |
| `centroid_vx`, `centroid_vy` | is the team moving forward or back |
| `width`, `length` | spread of the outfield players |
| `line_height` | deepest outfield player (defensive line) |
| `mean_speed`, `n_sprinting` | intensity |
| `pressing` | how many players are closing in on one point |
| `n_tracked` | players visible this frame |

## 4. Whole game

| feature | meaning |
|---|---|
| `mean_speed_all` | drops sharply at stoppages |
| `speed_drop` | change in `mean_speed_all` over the last 2 s |
| `ball_stationary_s` | how long the ball has been still |
| `at_restart_spot` | ball near touchline / corner / 6-yard box / centre spot |
| `wall` | 3+ players in a line ~9 m from a stationary ball |
| `tag_set_piece`, `tag_set_piece_type` | Smart Tagging set piece active (and type) |
| `tag_team` | Smart Tagging phase team, H / A / none (weak hint) |

## 5. Context

| feature | meaning |
|---|---|
| `half` | 1 or 2 |
| `attack_dir_H` | which goal Barcelona attacks this half |
| `t_in_half_s` | time since kickoff of the half |

## What the labeller uses vs what the model uses

The deterministic labeller decides from groups 1, 2 and 4 (ball and dead-ball evidence).
The model is trained on its confident labels and **leans on group 3**, so it learns what
possession looks like without the ball. That is what lets it help in the gaps.
