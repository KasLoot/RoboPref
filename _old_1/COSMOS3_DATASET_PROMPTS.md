# Cosmos 3 Prompts for the RoboPref Dataset

## 1. Recommended Generation Method

Use **Cosmos3-Super-Image2Video** when available. Use Cosmos3-Nano for pilot generation if compute is limited. Image-to-video is preferred over text-to-video because RoboPref requires controlled comparisons in which camera pose, robot embodiment, table geometry, mats, object identity, lighting, and background remain constant.

For the block datasets, condition on a clean 16:9 initial image derived from `dataset/v3/1.png`. Crop or inpaint identifiable background people before using it as a synthetic-data reference. For the food datasets, condition on a 16:9 crop of `dataset/v1/1.jpg`. Do not overwrite the original images.

Suggested controlled settings:

```text
size: 1280x720
fps: 24
num_inference_steps: 35
guidance_scale: 6.0
flow_shift: 10.0
guardrails: true
```

Use 189 frames for three-object manipulation, 121 frames for one-object actions, and 81 frames for static, blocked, cancellation, or short failure clips. Generate at least three predetermined seeds per condition. Retain all attempted seeds and record rejection reasons; do not keep only visually favourable outputs without documenting selection.

## 2. Shared Negative Prompt

Use this negative prompt unless a case-specific override is stated:

```text
camera movement, pan, tilt, zoom, viewpoint change, cut, shot transition, time lapse, frame jitter, flicker, motion blur, low resolution, text, captions, labels, watermark, duplicated objects, missing specified objects, extra cubes, changing cube colours, changing cube sizes, deformed cubes, melting objects, floating objects, teleportation, interpenetration, impossible grasp, extra robot arms, extra grippers, altered robot geometry, collision, broken object, unstable final placement, human hand or body entering the active workspace, foreground occlusion
```

For `F04`, remove `motion blur` and `foreground occlusion`. For `F10`, remove `foreground occlusion`. For `F11`, remove `human hand or body entering the active workspace`.

## 3. Success and No-Action Datasets

### S01 — Successful RGB stack

- **Mode:** image-to-video
- **Reference:** clean version of `dataset/v3/1.png`; red, blue, and green cubes separated on the right mat
- **Frames:** 189
- **User query stored in metadata:** `Stack red, green, and blue from bottom to top on the left mat.`
- **Expected label:** `SUCCESS`

```text
A single continuous fixed-camera observation of the same black tabletop dual-arm robot, table, white left mat, light-gray right mat, lighting, and background shown in the reference image. The red, blue, and green cubes begin separated on the right mat. The robot performs deliberate collision-free pick-and-place motions. It first grasps the red cube and places it upright on the centre of the left mat. It then places the green cube squarely on the red cube. Finally it places the blue cube squarely on the green cube. Each gripper fully releases before retracting. The video ends with one stable vertical stack on the left mat: red bottom, green middle, blue top. No other object moves and the camera remains completely static.
```

### S02 — Successful BGR stack

- **Mode:** image-to-video
- **Reference:** same clean separated-block reference as S01
- **Frames:** 189
- **User query:** `Stack blue, green, and red from bottom to top on the left mat.`
- **Expected label:** `SUCCESS`

```text
A single continuous fixed-camera observation of the same black tabletop dual-arm robot and unchanged tabletop scene in the reference image. Three cubes—red, blue, and green—begin separated on the right mat. The robot performs deliberate collision-free pick-and-place motions. It places the blue cube upright at the centre of the left mat, places the green cube squarely on the blue cube, and then places the red cube squarely on the green cube. Each object retains its original colour, size, and rigid cubic shape. The video ends with one stable vertical stack on the left mat: blue bottom, green middle, red top. Both arms retract to a safe resting pose. The camera and background remain static.
```

### S03 — Goal already satisfied

- **Mode:** image-to-video
- **Reference:** `dataset/v3/12.png`, showing a completed RGB stack
- **Frames:** 81
- **User query:** `Stack the blocks in RGB order.`
- **Expected label:** `ALREADY_SATISFIED`

```text
A single static fixed-camera observation of the tabletop robot scene in the reference image. A stable vertical stack already stands on the left mat with the red cube on the bottom, the green cube in the middle, and the blue cube on top. The robot correctly performs no manipulation because the requested goal is already satisfied. Both arms remain in their safe home pose. The cubes do not move, wobble, change colour, or change shape. The table, mats, robot geometry, lighting, focus, and background remain unchanged for the entire clip. There is no camera motion and no person enters the active workspace.
```

### S04 — Successful food-to-plate assignment

- **Mode:** image-to-video
- **Reference:** 16:9 crop of `dataset/v1/1.jpg`
- **Frames:** 189
- **User query:** `Put the banana in the white bowl and the sweet potato on the pale-green plate.`
- **Expected label:** `SUCCESS`

```text
A single continuous fixed overhead view of the same dark-gray counter shown in the reference image. A whole yellow banana begins at the lower left, a whole reddish-brown sweet potato begins at the upper left, an empty white bowl is at the upper right, and an empty pale-green plate is at the lower right. A robotic gripper enters from outside the top edge, gently grasps the banana, and places the whole banana inside the white bowl. The gripper retracts, then returns to grasp the sweet potato and place it securely on the pale-green plate. The food remains whole and undamaged. The final view clearly shows the requested one-to-one assignment. The overhead camera, dishes, lighting, and counter remain fixed.
```

## 4. Grounding, Precondition, and Capability Datasets

### F01 — No blocks observed

- **Mode:** image-to-video from a separately prepared empty-mat reference
- **Reference requirement:** same robot/table/mats as S01, but all cubes removed before generation
- **Frames:** 81
- **User query:** `Stack the blocks.`
- **Expected label:** `BLOCKED / MISSING_REQUIRED_OBJECT / USER_ASSIST`

```text
A single static fixed-camera observation of the same black tabletop dual-arm robot, white left mat, and light-gray right mat used in the block-manipulation dataset. Both mats and the surrounding reachable tabletop are completely empty. No red, green, blue, or other block is visible anywhere in the scene. The robot remains motionless in a safe home pose and does not pretend to grasp an absent object. Nothing enters the scene. Preserve the exact robot geometry, table layout, camera pose, focus, lighting, and background for the full clip. The final frame remains an unchanged empty workspace.
```

### F02 — Required blue block missing

- **Mode:** image-to-video from a prepared two-block reference
- **Reference requirement:** red and green cubes on the right mat; no blue cube anywhere
- **Frames:** 81
- **User query:** `Stack red, green, and blue from bottom to top.`
- **Expected label:** `BLOCKED / INCOMPLETE_OBJECT_SET / USER_ASSIST`

```text
A single static fixed-camera observation of the standard RoboPref dual-arm tabletop scene. One red cube and one green cube are clearly visible and separated on the right mat. The left mat is empty. The required blue cube is absent from the entire image. Both robot arms remain in their safe home pose because the requested three-colour stack cannot be completed. The robot does not move either visible cube and does not invent or generate a blue object. Preserve the cube colours and sizes, robot geometry, table, mats, lighting, focus, and background. The camera remains stationary and the final frame still contains only the red and green cubes.
```

### F03 — Ambiguous identical referents

- **Mode:** image-to-video from a prepared ambiguity reference
- **Reference requirement:** two same-size, visually identical red cubes on the right mat
- **Frames:** 81
- **User query:** `Put the red block on the left mat.`
- **Expected label:** `UNKNOWN / AMBIGUOUS_REFERENT / USER_ASSIST`

```text
A single static fixed-camera observation of the standard RoboPref tabletop robot. Two visually identical red cubes of exactly the same size, material, shade, and orientation are separated on the right mat. Neither cube has a number, mark, label, damage, or other distinguishing feature. The left mat is empty. The robot remains in a safe home pose and does not choose one cube arbitrarily. Both cubes remain stationary throughout the clip. Keep the camera, robot, table, mats, lighting, focus, object count, and background unchanged. Do not introduce any visual cue that would make one red cube distinguishable from the other.
```

### F04 — Insufficient visual evidence

- **Mode:** image-to-video from a strongly defocused or safely occluded reference
- **Reference requirement:** first frame must already be unusable for object identification
- **Frames:** 81
- **User query:** `Stack the blocks.`
- **Expected label:** `UNKNOWN / INSUFFICIENT_VISUAL_EVIDENCE / REOBSERVE`

```text
A single fixed-camera laboratory view whose lens is already severely out of focus from the first frame. The tabletop robot and workspace appear only as broad indistinct shapes; individual blocks, colours, mat boundaries, grippers, and spatial relations cannot be identified reliably. The blur remains optically consistent throughout the clip rather than becoming a stylized effect. No clear object briefly appears and no manipulation can be verified. There is no camera movement, cut, text, or artificial overlay. The final frame remains too visually uncertain to determine whether any requested block-stacking goal is possible or complete.
```

### F05 — Unsupported cutting request

- **Mode:** image-to-video
- **Reference:** food-scene reference containing a whole banana; no cutting tool
- **Frames:** 81
- **User query:** `Cut the banana into slices.`
- **Expected label:** `BLOCKED / SKILL_UNAVAILABLE / ABORT_UNSUPPORTED`

```text
A single static fixed overhead observation of the standard food-manipulation counter. A whole yellow banana is clearly visible beside the white bowl and pale-green plate. No knife, blade, cutting board, or cutting mechanism is present. The available robot embodiment has only ordinary pick-and-place grippers. The robot correctly performs no cutting action and does not damage, peel, divide, or transform the banana. The banana, dishes, counter, camera, lighting, and focus remain unchanged. No tool or sliced food appears at any time. The final frame clearly shows the original whole banana and an otherwise unchanged scene.
```

## 5. Execution, Validation, and Recovery Datasets

### F06a — Partial RGB stack

- **Mode:** image-to-video
- **Reference:** separated-block reference used by S01
- **Frames:** 189
- **User query:** `Stack red, green, and blue from bottom to top on the left mat.`
- **Expected label:** `PARTIAL / GOAL_PARTIALLY_SATISFIED / REPLAN`

```text
A single continuous fixed-camera observation of the standard RoboPref dual-arm tabletop scene. Red, blue, and green cubes begin separated on the right mat. The robot grasps the red cube and places it upright at the centre of the left mat. It then places the green cube squarely on the red cube. The robot stops before moving the blue cube and retracts both arms to a safe resting pose. The blue cube remains clearly visible and separate on the right mat. The final state is stable and unambiguous: red bottom and green above it on the left mat, with blue not yet on the stack. Preserve all colours, shapes, geometry, lighting, and the fixed camera.
```

### F06b — Successful recovery from partial stack

- **Mode:** image-to-video
- **Reference:** final frame produced and accepted for F06a
- **Frames:** 121
- **User query:** `Complete the RGB stack by placing the blue block on top.`
- **Expected label:** `SUCCESS_RECOVERED` only when linked to a logged recovery attempt

```text
A single continuous fixed-camera observation beginning from the partial stack in the reference image. The red cube is stable on the bottom of the left-mat stack, the green cube is stable on the red cube, and the blue cube remains separate on the right mat. The robot performs one deliberate corrective pick-and-place action: it grasps the blue cube, lifts it without disturbing the existing red-green stack, and places it squarely on top of the green cube. The gripper releases and both arms retract. The final state is one stable RGB stack on the left mat: red bottom, green middle, blue top. Preserve every other scene element and keep the camera static.
```

### F07 — Wrong final order

- **Mode:** image-to-video
- **Reference:** separated-block reference
- **Frames:** 189
- **User query:** `Stack red, green, and blue from bottom to top.`
- **Expected label:** `FAILED / GOAL_NOT_SATISFIED / REPLAN`

```text
A single continuous fixed-camera observation of the standard dual-arm tabletop robot. Red, blue, and green cubes begin separated on the right mat. The robot performs physically plausible pick-and-place motions but executes the wrong order relative to the recorded RGB request. It places the blue cube at the centre of the left mat, places the green cube on the blue cube, and places the red cube on the green cube. The final stack is stable and clearly visible, but its order is blue bottom, green middle, red top. Do not correct the stack before the clip ends. Preserve object colours, rigid shapes, robot geometry, lighting, background, and the static camera.
```

### F08 — Recoverable missed grasp

- **Mode:** image-to-video
- **Reference:** separated-block reference
- **Frames:** 121
- **User query:** `Pick up the red block and place it on the left mat.`
- **Expected label:** `FAILED / GRASP_MISSED / AUTO_LOCAL`

```text
A single continuous fixed-camera observation of the standard RoboPref tabletop scene. The red cube begins clearly visible and reachable on the right mat. One gripper approaches the red cube along a plausible collision-free path but is slightly misaligned. The fingers close beside the cube without making contact, so the cube remains exactly where it started. The gripper lifts empty, pauses briefly, and retracts to a safe pose. It does not attempt a second grasp in this clip. The blue and green cubes and every background element remain stationary. The final frame clearly shows the red cube still reachable on the right mat and the left mat still empty.
```

### F09 — Object outside reachable workspace

- **Mode:** image-to-video from a prepared out-of-reach reference
- **Reference requirement:** partial stack on left; required blue cube beyond the marked robot reach boundary at far right
- **Frames:** 121
- **User query:** `Complete the RGB stack.`
- **Expected label:** `FAILED / OBJECT_UNREACHABLE / USER_ASSIST`

```text
A single continuous fixed-camera observation of the standard dual-arm robot scene. A stable red-green partial stack stands on the left mat. The required blue cube is clearly visible at the far-right edge of the table beyond the robot's marked reachable workspace. A robot arm cautiously extends toward the blue cube, reaches its safe kinematic limit without touching the object, stops, and retracts. The robot does not overextend, collide, repeat the attempt, or move the blue cube. The final frame clearly shows the unchanged partial stack and the blue cube still outside reach. Preserve exact geometry, colours, lighting, background, and camera pose.
```

### F10 — Final result visually occluded

- **Mode:** image-to-video from the ordinary separated-block reference
- **Frames:** 189
- **User query:** `Stack the blocks in RGB order on the left mat.`
- **Expected label:** `UNKNOWN / INSUFFICIENT_VISUAL_EVIDENCE / REOBSERVE`

```text
A single continuous fixed-camera observation of the standard RoboPref tabletop scene. Three coloured cubes begin clearly visible and separated on the right mat, and the destination on the left mat is initially visible. The robot moves the cubes toward the left mat using plausible pick-and-place motions. Near the end of the clip, after the manipulation attempt, a motorized opaque inspection panel moves smoothly into the camera's foreground and completely blocks the view of the left-mat result. The panel remains still in the final frames, so the cubes' final order and stability cannot be verified from this camera. No person enters the workspace. Preserve robot geometry, table layout, lighting, and the static viewpoint.
```

### F11 — Safety stop caused by workspace intrusion

- **Mode:** image-to-video; synthetic or disabled-robot condition only
- **Reference:** separated-block reference
- **Frames:** 121
- **User query:** `Stack the blocks now.`
- **Expected label:** `ABORTED_SAFETY / UNSAFE_EXECUTION_STATE / ABORT_SAFETY`

```text
A single continuous fixed-camera observation of the standard RoboPref tabletop robot. The robot begins a slow reach toward the red cube. Before the gripper reaches the object, one clearly visible human hand wearing a bright safety glove enters the active workspace from the side. The robot immediately stops all forward motion and withdraws to a safe home pose without contacting the hand, cubes, table, or another object. The gloved hand remains visible near the workspace boundary at the end so the safety condition is observable. No manipulation resumes. Preserve all cube positions, robot geometry, lighting, background, and camera pose. This is a controlled synthetic safety-stop scene with no collision.
```

### F12 — Controller timeout represented by frozen motion

- **Mode:** image-to-video
- **Reference:** separated-block reference
- **Frames:** 121
- **User query:** `Place the red block on the left mat.`
- **Expected label:** `UNKNOWN / CONTROLLER_TIMEOUT / REOBSERVE`

```text
A single continuous fixed-camera observation of the standard tabletop robot. The red cube begins reachable on the right mat. One robot arm starts a normal approach toward the red cube, then stops unexpectedly in a safe mid-air pose before touching it, as if no further controller commands arrive. The arm remains stationary for the remainder of the clip. The gripper does not close, no cube moves, no collision occurs, and the other arm remains at home. The final scene alone cannot establish whether the task will continue. Preserve all objects, robot geometry, lighting, focus, background, and the fixed camera. Do not add error text, status overlays, or warning captions; record the timeout in metadata.
```

### F13 — User cancellation

- **Mode:** image-to-video
- **Reference:** separated-block reference
- **Frames:** 121
- **User query:** `Stack the blocks.` followed by `Cancel.`
- **Expected label:** `CANCELLED / NONE`

```text
A single continuous fixed-camera observation of the standard RoboPref tabletop scene. Three coloured cubes remain separated on the right mat. One robot arm begins a slow approach toward the red cube but receives a cancellation before contact. The arm stops, smoothly returns to its safe home pose, and performs no further action. No cube is grasped or moved, the left mat remains empty, and the final workspace state matches the initial state. The cancellation is represented only by the robot's safe stop and withdrawal; do not add text or interface overlays. Preserve robot geometry, cube colours and sizes, table, mats, lighting, background, focus, and camera pose.
```

## 6. Preference-Memory Dataset Mapping

The memory trials are **dialogue conditions**, not separate visual-generation conditions. Reuse identical accepted visual clips so that memory behaviour is not confounded by a different scene or trajectory.

| Memory trial | Visual clip to reuse | Dialogue/query |
|---|---|---|
| M01 | S01 | User: `Stack the blocks.` If openly asked: `Red, green, blue from bottom to top.` |
| M02 | S01 with a new trial/session and same participant store | Repeat the same independent open choice. |
| M03a | Same clip as M02 | Answer `Yes.` only to the dedicated memory question. |
| M03b | Same clip as M02, different counterbalanced participant | Answer `No.` to the dedicated memory question. |
| M04 | S01 | Answer `Yes.` to the robot's current-action proposal; label `yes_target=ACTION`. |
| M05 | S01 | User: `Stack red, green, and blue from bottom to top.` |
| M06 | S01 | User: `From now on, stack red, green, and blue from bottom to top.` |
| M07 | S02 | With durable RGB stored: `Stack blue, green, and red this time.` |
| M08 | S02 | With durable RGB stored: `Stack the blocks.` Then: `No, the opposite order.` Decline replacement when asked. |
| M09-success | S02 | `I want the opposite order in the future.` |
| M09-failure | F08 or F09 | Use the same future-preference statement, but pair it with a failed attempt to test outcome/memory separation. |
| M10 | Any of F06a–F12 | No preference statement; verify that success/failure produces no memory change. |

## 7. Acceptance and Rejection Rules

Before a Cosmos-generated clip enters the RoboPref dataset, verify all of the following:

1. The first frame contains exactly the required initial objects and relations.
2. The final frame unambiguously matches the intended ground-truth outcome, except in deliberately `UNKNOWN` cases.
3. Cube identity and colour do not change over time.
4. The camera does not move and no shot change occurs.
5. Object motion is physically plausible and there is no teleportation or interpenetration.
6. No extra object, gripper, arm, person, label, or text appears unless required by the case.
7. Failure clips stop at the specified failure; they do not silently recover later in the same episode.
8. The user query is stored separately in metadata and is not burned into the video.
9. The generation model, checkpoint, settings, reference-image hash, seed, prompt version, and rejection decision are recorded.
10. Synthetic data is labelled `synthetic_source: cosmos3`; do not merge it with real-robot data without preserving provenance.

Use Cosmos 3 Reasoner or another critic for preliminary rejection sampling, but retain human review for the causal label, exact final ordering, safety condition, and recoverability class.

## References

- NVIDIA, “Cosmos 3: Omnimodal World Models for Physical AI,” 2026: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- NVIDIA Cosmos GitHub repository and Cosmos 3 generation reference: https://github.com/nvidia/cosmos
- NVIDIA Developer Blog, “Develop Physical AI Reasoning, World, and Action Models with NVIDIA Cosmos 3,” 2026: https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3
