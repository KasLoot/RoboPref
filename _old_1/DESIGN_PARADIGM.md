# Consent-Gated Preference Learning and Task Assurance for Interactive VLA Robot Systems

## Abstract

This document specifies a human-centred control paradigm for an interactive robotic manipulation system composed of a Human–Robot Interaction (HRI) agent, a persistent preference store, a vision-language planner, a Vision-Language-Action (VLA) executor, and an independent outcome validator. The design addresses two coupled hazards: learning a durable preference from a statement that only authorises the current action, and treating a plausible plan or attempted action as evidence of task success. The proposed system separates current-task intent, candidate preference evidence, durable preference memory, and task-outcome history. Durable memory is written only after explicit future-oriented language or an affirmative answer to a dedicated memory question. Repeated unmarked choices remain candidates; in the prototype, two independent matching choices trigger a request for consent rather than automatic promotion. Task execution is governed by a task-assurance loop that checks grounding, preconditions, affordances, plan validity, execution evidence, and goal postconditions. Failures are represented structurally and assigned to bounded recovery classes: re-observation, local retry, replanning, user assistance, unsupported-task abort, and immediate safety abort. The design therefore makes adaptation inspectable and correctable while distinguishing execution failure from preference evidence.

## 1. Scope and Research Position

The target application is a tabletop service or laboratory robot that receives natural-language manipulation requests and scene images. The current RoboPref prototype operates on recorded episodes: an initial frame is provided to the HRI agent and planner, and a final frame is provided to the validator. It does not yet command physical actuators. Consequently, the implemented task-assurance layer can determine whether planning should proceed, classify the observed final outcome, and recommend a recovery action, but it must not claim to have executed a physical retry. A future VLA or robot controller may consume the same structured recovery decision.

The central design claim is that **preference adaptation and task assurance must be coupled at the interaction level but separated in state and evidence**. A robot may learn that a user explicitly wants a particular block order in future even when the present attempt fails. Conversely, a successful trajectory does not demonstrate that an incidental task parameter is a durable preference.

The work is guided by mixed-initiative interaction: initiative should be allocated according to the expected value of asking, acting, or deferring rather than by a blanket rule to always ask or never ask [1]. Human-AI guidance similarly recommends cautious adaptation, support for correction, disclosure of consequential actions, and user control over learned behaviour [2]. Preference elicitation has a measurable interaction or “bother” cost, motivating selective questions [3], while explicit elicitation should provide sufficient expressive benefit to justify user effort [4].

## 2. Design Objectives and Requirements

The design has eight normative requirements.

| ID | Requirement |
|---|---|
| R1 | The system shall distinguish approval of a proposed current action from consent to store a future preference. |
| R2 | A fully specified current command shall not, by itself, be persisted as a preference. |
| R3 | Repeated implicit or unmarked evidence shall never silently become durable memory. |
| R4 | The current explicit instruction shall control the current task even when it conflicts with memory; replacing the saved default requires consent. |
| R5 | Planner feasibility and final task success shall be checked against observable preconditions and postconditions. |
| R6 | Failure responses shall be proportional to recoverability, bounded in retry count, and immediately conservative for safety events. |
| R7 | Preference evidence, task outcomes, and capability/reliability evidence shall remain logically separate. |
| R8 | Every user-visible claim of success, failure, memory use, or memory update shall be traceable to structured evidence. |

## 3. Evidence Base and Design Rationale

Horvitz’s mixed-initiative principles frame interaction as a decision problem involving uncertainty, the cost of interruption, and the value of obtaining information [1]. RoboPref applies this principle twice: it asks a task-clarification question only when ambiguity affects the outcome, and it asks a memory question only when evidence is sufficiently useful to justify interruption.

Amershi et al. derived 18 human-AI interaction guidelines through multiple validation phases involving 49 practitioners and 20 AI-infused products [2]. Particularly relevant principles are that an AI system should learn from user behaviour, update cautiously, support efficient correction, convey the consequences of actions, and provide global controls. RoboPref therefore exposes memory state, treats candidates as reversible evidence, and requires explicit consent at the point where persistence becomes consequential.

Gucsi et al. formulate social-robot preference elicitation under a user-annoyance budget [3]. Their result supports a cost-aware question policy, but does not establish a universal number of repeated observations after which a preference becomes valid. Pommeranz et al. likewise show that users may accept additional elicitation effort when the feedback mechanism is expressive enough to warrant that effort [4]. Maroto-Gómez et al. report a longitudinal study with 24 participants in which combined explicit and implicit feedback produced better adaptive values than either source alone [5]. These results motivate combining candidate observations with a short explicit confirmation.

The prototype uses **two independent matching user choices as the trigger to ask** whether the choice should be remembered. This value is a conservative engineering prior, not an empirically universal threshold. It minimises early interruptions while giving the system an opportunity to confirm a potentially stable pattern. It must be treated as a tunable parameter and evaluated by measuring false-promotion rate, missed-preference rate, question burden, correction rate, and user trust.

For robot failures, Honig and Oron-Gilad’s review of 52 HRI studies distinguishes technical and interaction failures and emphasises failure communication, human comprehension, and resolution [6]. Their account motivates explicit stages, causes, recoverability, and user-facing next steps. SayCan demonstrates the importance of grounding language-level usefulness in the robot’s estimated ability to execute a skill in the current state [7]. Runtime monitoring is also necessary: SAFE shows that task-general VLA failure detection can support stopping, backtracking, or asking for help [8], while recent hierarchical VLA recovery work separates global planning from local corrective action [9]. Behaviour-tree research provides an established model in which preconditions gate action, postconditions determine completion, and execution is continuously monitored [10].

## 4. Unified Architecture

The proposed control flow is:

```text
User query
   |
   v
HRI grounding and ambiguity resolution
   |---- missing/unsafe/unsupported ----> structured REPORT
   v
Confirmed current-task contract
   |                         \
   |                          -> preference evidence curator
   v                               | candidate only / explicit consent
Scene precondition and affordance check
   |---- blocked ------------> recovery policy
   v
Structured plan
   v
VLA/robot execution + independent monitoring
   v
Postcondition validator
   |---- incomplete/unknown --> recovery policy
   v
Verified outcome report
```

Three stores are conceptually distinct:

1. **Preference store:** user-specific candidate and durable defaults, their scope, and direct evidence.
2. **Task-outcome log:** per-attempt expected state, observed state, outcome, failure code, and recovery action.
3. **Capability model:** aggregate success likelihood for skills, objects, contexts, and embodiments.

The present implementation provides the preference store and returns a structured task-assurance result. Persistence of outcome and capability histories is reserved for a later executor integration. The architecture forbids copying task success into preference evidence or treating a preferred action as proof of robot capability.

## 5. Preference-Memory Paradigm

### 5.1 Evidence classes

Each potential preference is assigned an evidence origin.

| Origin | Example | Memory effect |
|---|---|---|
| `explicit_preference` | “From now on, use red–green–blue.” | Durable immediately. |
| `memory_confirmation` | Robot: “Remember RGB as your default?” User: “Yes.” | Confirm selected candidate; durable. |
| `open_preference_answer` | Robot: “Which order would you like?” User: “RGB.” | Add/reinforce candidate. |
| `user_initiated_override` | “Use BGR instead.” without future scope | Current task wins; candidate evidence only. |
| `assistant_plan_confirmation` | Robot proposes RGB; user says “Yes.” | Current action approval only; no preference evidence. |
| `current_task` | “Stack RGB.” | Current task constraint only; no preference evidence. |
| `system_inference` | Successful execution or validator output | Never preference evidence. |

An answer is independent only if it constitutes a new user choice rather than a repeated rendering of the same utterance, a paraphrase generated by the system, or an acknowledgement of an action proposal. Independence should ultimately be session- and context-aware; the prototype records the curator’s independence claim and distinct interaction session identifiers.

### 5.2 Candidate and durable states

For a preference identity \(p=(user, task\_type, key, context, scope)\), the state machine is:

```text
no record --unmarked open choice--> candidate
candidate --matching independent choice--> candidate + confirmation pending
candidate --explicit memory yes--> durable
candidate --explicit future language--> durable
candidate --memory no--> candidate, confirmation closed
durable --conflicting current command--> execute override + replacement question
durable --replacement yes--> old retracted, new durable
durable --replacement no--> retain old durable; current override remains one-off
```

Evidence count is not authority. No value of the count alone permits transition to durable. The two-observation rule only schedules a confirmation question.

### 5.3 Interpretation of “yes”

Affirmations are typed by the immediately preceding question:

- “Should I execute this RGB stack?” → action confirmation; do not update memory.
- “Should I remember RGB as your default for future stacks?” → memory confirmation; promote the selected candidate.
- A bare “yes” with missing or uncertain conversational antecedent → ask what is being confirmed; do not persist.

This policy prevents an assistant-generated proposal from manufacturing its own preference evidence. It also makes the consequence of the user’s answer visible before durable storage.

### 5.4 Conflict policy

If a current instruction conflicts with a durable default, the current instruction always controls execution. A conflict does not automatically retract or replace memory. After the task resolution is complete, the system may ask one concise question: “You chose BGR, which differs from your saved RGB default. Should BGR replace it for future block stacks?” Failure or success of the current attempt does not change the answer’s semantic force.

### 5.5 Question burden

Memory confirmation is deferred until after the current task has been resolved so that it cannot be mistaken for task authorisation. Questions are suppressed when evidence is one-off, when the same confirmation has already been declined without new evidence, or when higher-priority safety/user-assistance communication is required. For high-frequency use, the trigger should later become a cost-sensitive policy using expected error reduction, interruption cost, task stakes, and evidence stability rather than a fixed count [1,3].

## 6. Task Contract, Assurance, and Recovery

### 6.1 Task contract

The HRI agent produces a user-approved task contract containing grounded objects, source and destination regions, required assignment/order, constraints, assumptions, and goal conditions. The planner must not weaken or silently elaborate this contract. The validator evaluates the same goal conditions.

### 6.2 Planning statuses

Before execution, the planner returns exactly one status:

- `READY`: visible preconditions and available skills support a concrete plan.
- `ALREADY_SATISFIED`: the observable scene already meets all goal conditions.
- `BLOCKED`: a required object, target, relation, or state is absent or ambiguous.
- `UNSUPPORTED`: the task is outside the declared skill/embodiment capability.
- `UNSAFE`: execution violates a safety constraint.
- `UNKNOWN`: evidence is insufficient or planner output is invalid.

For example, “stack the blocks” with no visible blocks is `BLOCKED`, code `MISSING_REQUIRED_OBJECT`, recoverability `USER_ASSIST`. It must not be dispatched to the VLA.

### 6.3 Outcome statuses

After execution, task assurance reports one of:

- `SUCCESS`: all observable goal conditions are satisfied.
- `SUCCESS_RECOVERED`: satisfied after at least one recorded recovery attempt.
- `ALREADY_SATISFIED`: no action was required.
- `PARTIAL`: observable progress exists but one or more goal conditions remain unmet.
- `BLOCKED`: execution cannot start or continue until an external precondition changes.
- `FAILED`: the attempt ended with sufficiently observed unmet goals.
- `ABORTED_SAFETY`: stopped because continued action could be unsafe.
- `CANCELLED`: stopped by the user.
- `UNKNOWN`: the system cannot reliably determine the state or outcome.

`UNKNOWN` is not failure and never success. It triggers re-observation when safe and available.

### 6.4 Failure record

Every non-success assurance result contains:

```json
{
  "stage": "GROUNDING | PRECONDITION | PLANNING | EXECUTION | VALIDATION | SAFETY",
  "code": "stable_machine_readable_code",
  "expected": "required condition",
  "observed": "available evidence",
  "confidence": 0.0,
  "severity": "LOW | MEDIUM | HIGH | CRITICAL",
  "recoverability": "REOBSERVE | AUTO_LOCAL | REPLAN | USER_ASSIST | ABORT_UNSUPPORTED | ABORT_SAFETY",
  "safe_state": "what the system knows about the robot/workspace",
  "attempts": 0,
  "next_action": "bounded recommended action",
  "user_message": "brief truthful explanation",
  "memory_effect": "NONE"
}
```

### 6.5 Bounded recovery policy

The default prototype budget is one local retry after re-observation and one alternate replan. These values are engineering safeguards, not universal constants.

1. `REOBSERVE`: acquire a new frame or viewpoint once; if uncertainty remains, request assistance.
2. `AUTO_LOCAL`: place the robot in a known safe state and retry the same short-horizon skill once.
3. `REPLAN`: reassess the scene and generate one materially different plan.
4. `USER_ASSIST`: explain the missing precondition and request one concrete intervention.
5. `ABORT_UNSUPPORTED`: do not retry; explain the capability boundary and offer a feasible alternative.
6. `ABORT_SAFETY`: stop immediately, do not automatically resume, and require an explicit safety-clear signal.

The current episode-based implementation emits these recommendations but does not execute them. A physical controller must acknowledge a completed safe-state transition and log each attempt before `SUCCESS_RECOVERED` can be claimed.

## 7. Scenario Catalogue

The following catalogue spans both success and failure conditions.

| Stage | Scenario | Classification | Unified response |
|---|---|---|---|
| HRI | Unique, feasible instruction | Proceed | Create contract and plan. |
| HRI | Ambiguous assignment/order | `ASK`/`CONFIRM` | Ask one outcome-relevant question; type the reply by question purpose. |
| HRI | User changes command mid-dialogue | Restart | Discard unresolved contract, retain no preference from abandoned proposal. |
| HRI | User cancels | `CANCELLED` | Stop; no memory update unless cancellation explicitly states a future preference. |
| Grounding | No requested blocks visible | `BLOCKED/MISSING_REQUIRED_OBJECT` | Do not invoke VLA; ask user to place blocks or revise task. |
| Grounding | One of several required blocks missing | `BLOCKED/INCOMPLETE_OBJECT_SET` | Name visible/missing evidence; request assistance. |
| Grounding | Duplicate visually indistinguishable objects | `UNKNOWN/AMBIGUOUS_REFERENT` | Request a distinguishing description or new viewpoint. |
| Perception | Occlusion, blur, darkness, stale frame | `UNKNOWN/INSUFFICIENT_VISUAL_EVIDENCE` | Re-observe once, then request assistance. |
| Planning | Goal already satisfied | `ALREADY_SATISFIED` | Report no action required; validate if uncertainty remains. |
| Planning | Planner emits malformed output | `UNKNOWN/MALFORMED_PLANNER_OUTPUT` | Retry planning once; never dispatch unparsed text. |
| Planning | Empty plan for incomplete task | `BLOCKED/NO_FEASIBLE_PLAN` | Replan once or request assistance based on cause. |
| Capability | Object is visible but no suitable skill exists | `UNSUPPORTED/SKILL_UNAVAILABLE` | Abort unsupported; offer feasible variant. |
| Safety | Human hand enters workspace or collision risk arises | `ABORTED_SAFETY` | Immediate stop; require explicit clearance before a new attempt. |
| Execution | Grasp misses but object remains reachable | `FAILED` or `PARTIAL`, `AUTO_LOCAL` | Re-observe, safe reset, retry once. |
| Execution | Object slips into a new reachable pose | `PARTIAL`, `REPLAN` | Update state and generate alternate plan. |
| Execution | Object leaves reachable workspace or breaks | `BLOCKED`, `USER_ASSIST` | Stop, report state, request concrete assistance. |
| Execution | Controller/communication timeout | `UNKNOWN` | Stop commands, verify robot state, retry communication once or abort. |
| Validation | All goal relations visible | `SUCCESS` | Report verified completion. |
| Validation | Some objects correctly placed | `PARTIAL` | Preserve correct work; plan only unmet postconditions. |
| Validation | Required result is occluded | `UNKNOWN` | Obtain another observation; do not declare failure or success. |
| Validation | Wrong order/target | `FAILED`, usually `REPLAN` | Explain discrepancy and produce bounded corrective plan. |
| Memory | User approves an action proposal | No preference | Execute only. |
| Memory | User explicitly specifies future default | Durable | Store independently of task outcome. |
| Memory | Two independent matching open choices | Candidate + prompt | Ask whether to remember; do not auto-promote. |
| Memory | Current choice conflicts with durable default | Execute override + prompt | Ask whether to replace default; retain old value unless confirmed. |
| Memory | Task succeeds/fails | No preference effect | Log outcome separately. |

## 8. Experimental and Evaluation Paradigm

Evaluation should separate interaction correctness from manipulation correctness.

### 8.1 Preference-memory measures

- **False durable-write rate:** fraction of durable records not supported by explicit future scope or a memory-confirmation answer.
- **Missed durable-preference rate:** explicit future preferences not correctly stored.
- **Action-confirmation contamination:** proportion of plan-confirmation “yes” replies incorrectly recorded.
- **Question burden:** clarification and memory questions per completed command.
- **Replacement correctness:** whether conflicting current instructions execute while stored defaults change only with consent.
- **User correction rate and perceived control:** post-session measures of unexpected adaptation and ease of correction.

### 8.2 Task-assurance measures

- Precondition-block detection for missing/ambiguous objects.
- Task success, partial completion, and failure classification accuracy.
- False-success rate, which should be treated as the primary safety-sensitive error.
- Recovery selection accuracy and attempts per recovered success.
- Time to failure detection and time to safe state.
- Fraction of failures resolved autonomously, with user assistance, or by abort.

### 8.3 Ablations

Recommended comparisons are: immediate implicit persistence versus consent-gated persistence; fixed two-choice trigger versus cost-sensitive trigger; planner self-report alone versus independent postcondition validation; and unbounded retry versus the proposed bounded policy. The present unit-test suite is a software conformance test, not evidence of human-subject efficacy. A later HRI study should counterbalance conditions, obtain informed consent, and report the trigger value as a preregistered design parameter.

## 9. Traceability to the RoboPref Prototype

| Requirement | Intended implementation evidence |
|---|---|
| R1–R4 | `memory/store.py`, memory-curator prompt, HRI memory-confirmation handling, memory tests. |
| R5–R6 | `assurance/task_assurance.py`, planner and validator schemas, task-assurance scenario tests. |
| R7 | Preference operations accept only user evidence origins; task result has `memory_effect: NONE`. |
| R8 | Structured HRI, planner, validator, and assurance JSON; deterministic parsing tests. |

## 10. Limitations

The HRI, planner, memory curator, and validator remain probabilistic model components. Structured schemas and deterministic gates reduce but do not eliminate visual grounding errors. The validator currently receives one final image and may miss occluded relations; `UNKNOWN` must therefore remain a first-class outcome. Candidate independence is approximated from interaction metadata rather than formally inferred from causal user behaviour. The fixed two-choice prompt threshold requires empirical validation with target users and tasks. Finally, this prototype does not implement actuator-level emergency stopping, collision monitoring, or certified safety; those functions must reside in independent robot control and safety systems.

## References

[1] E. Horvitz, “Principles of Mixed-Initiative User Interfaces,” in *Proceedings of CHI ’99*, pp. 159–166, 1999. [Online]. Available: https://erichorvitz.com/chi99horvitz.pdf

[2] S. Amershi et al., “Guidelines for Human-AI Interaction,” in *Proceedings of CHI 2019*, Paper 3, 2019. doi: [10.1145/3290605.3300233](https://doi.org/10.1145/3290605.3300233).

[3] B. Gucsi, D. S. Tarapore, W. Yeoh, C. Amato, and L. Tran-Thanh, “To Ask or Not to Ask: A User Annoyance Aware Preference Elicitation Framework for Social Robots,” in *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 7935–7940, 2020. doi: [10.1109/IROS45743.2020.9341607](https://doi.org/10.1109/IROS45743.2020.9341607).

[4] A. Pommeranz, J. Broekens, P. Wiggers, W.-P. Brinkman, and C. M. Jonker, “Designing Interfaces for Explicit Preference Elicitation: A User-Centered Investigation of Preference Representation and Elicitation Process,” *User Modeling and User-Adapted Interaction*, vol. 22, pp. 357–397, 2012. doi: [10.1007/s11257-011-9116-6](https://doi.org/10.1007/s11257-011-9116-6).

[5] M. Maroto-Gómez, M. Á. Malfaz, J. C. Castillo, Á. Castro-González, and M. Á. Salichs, “Personalizing Activity Selection in Assistive Social Robots from Explicit and Implicit User Feedback,” *International Journal of Social Robotics*, 2024. doi: [10.1007/s12369-024-01124-2](https://doi.org/10.1007/s12369-024-01124-2).

[6] S. Honig and T. Oron-Gilad, “Understanding and Resolving Failures in Human-Robot Interaction: Literature Review and Model Development,” *Frontiers in Psychology*, vol. 9, art. 861, 2018. doi: [10.3389/fpsyg.2018.00861](https://doi.org/10.3389/fpsyg.2018.00861).

[7] M. Ahn et al., “Do As I Can, Not As I Say: Grounding Language in Robotic Affordances,” arXiv:2204.01691, 2022. [Online]. Available: https://say-can.github.io/

[8] Q. Gu et al., “SAFE: Multitask Failure Detection for Vision-Language-Action Models,” in *Advances in Neural Information Processing Systems 38*, 2025. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html

[9] X. Li, Y.-L. Li, Y. Wang, H. Wang, and S. Wang, “TCoT: Trajectory Chain-of-Thoughts for Robotic Manipulation with Failure Recovery in Vision-Language-Action Model,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 8, pp. 6486–6494, 2026. doi: [10.1609/aaai.v40i8.37577](https://doi.org/10.1609/aaai.v40i8.37577).

[10] M. Mayr et al., “Combining Decision Making and Dynamical Systems for Monitoring and Executing Manipulation Tasks,” *e & i Elektrotechnik und Informationstechnik*, vol. 137, pp. 316–322, 2020. doi: [10.1007/s00502-020-00816-7](https://doi.org/10.1007/s00502-020-00816-7).
