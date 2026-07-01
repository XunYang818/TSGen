"""
Full version prompts - includes all Constraints sections
"""

from langchain_core.prompts import ChatPromptTemplate

# ================================
# Planning Agent Prompts
# ================================

ACTIVITY_PLANNING_REQUIREMENTS = """
## Activity Requirements
Generate a detailed activity plan, consider comprehensively and meticulously to simulate real daily life:
1. Activities should align with the person's lifestyle and routines, covering all aspects of daily life: such as sleep, meals, work/study, entertainment, personal affairs, household chores, etc., as detailed and realistic as possible
2. Time should be reasonably allocated, covering a full day, with each activity having clear start and end times
3. Activities should match the house layout and fully utilize each room. For activities involving going out (such as work, taking out trash, shopping, walking, etc.), rooms do not need to be considered
4. Activities should reflect personal characteristics and habits

## Activity Constraints
1. **Time Continuity**: There should be no time gaps between activities (current activity end time matches next activity start time), no conflicts or gaps should exist
2. **Room Rationality**: Activities match room functions
3. **Life Realism**: Conform to real life rhythms and habits
4. **Personalization**: Reflect personal characteristics from resident profile
5. **Outing Completeness**: Outgoing activities (work, shopping, walking, visiting friends, etc.) must be complete closed loops from leaving home to returning home. Do not split into multiple plans like "Going out", "Working", "Returning home". Example: "Leaving home at 8:00 AM to go to work, working all day at the company, getting off work at 6:00 PM and returning home"
"""

PLANNING_PROMPT_TEMPLATE = """
You are a professional life planner, follow the following requirements to generate activity plans:

{activity_planning_requirements}

Today is May 25, 2025, weekend. 
Based on the following resident profile and house layout, plan a day (24 hours) of activity arrangements for this resident.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}
"""

PLANNING_VALIDATION_PROMPT_TEMPLATE = """
You are a professional life planner, follow the following requirements to validate activity plans:

{activity_planning_requirements}

## Validation Focus
1. Time Verification: Check if all activities are continuous time periods (current activity end time matches next activity start time), no conflicts or gaps should exist.
2. Room Rationality Verification: Check if all involved rooms exist in the house layout, there should be no rooms that don't exist in the house layout (except for going out).
3. Note: Only review the activity plan. Do not review resident profile and house layout, nor propose optimization suggestions for them.

Based on the following resident profile and house layout, verify the rationality of the activity plan.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Current Activity Plan
{activity_plans_json}
"""

PLANNING_CORRECTION_PROMPT_TEMPLATE = """
You are a professional life planner, follow the following requirements to correct activity plans:

{activity_planning_requirements}

Based on the following resident profile and house layout, and the issues identified in the validation results, regenerate a corrected activity plan.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Original Activity Plan
{original_activity_plans_json}

## Validation Issue Description
{correction_content}
"""

# ================================
# Event Agent Prompts
# ================================

EVENT_REQUIREMENTS = """
## Event Definition
An event is an intermediate level between activities and action steps, with the following characteristics:
1. **Purpose-oriented**: Each event has a clear purpose and meaning, being the process of completing a specific sub-goal
2. **Relatively Fixed Space**: Usually completed within one room, movement between rooms, or outdoor activities outside the house layout
3. **Time Continuity**: Events are internally continuous time periods with no interruptions
4. **Behavioral Completeness**: Events have clear beginnings, processes, and endings
5. **Cognitive Level**: More specific than activities, more abstract than actions, reflecting human intentions and decisions

## Event Requirements
1. **Clear Purpose**: Each event should have a clear purpose, such as "preparing breakfast ingredients", "cooking breakfast", "dining", "cleaning dishes"
2. **Space Rationality**: Conducted in specific rooms (except outdoor activities), conforming to room functions and activity nature
3. **Time Allocation**: Reasonably allocate time based on event complexity, considering resident behavioral habits
4. **Logical Sequence**: Events have reasonable logical relationships and time sequences
5. **Style Continuity**: Maintain consistent granularity and style with previously generated events
6. **Realistic Simulation**: Consider comprehensively and meticulously, simulate real human behavior patterns, including transitional behaviors and natural sequences. Each activity typically corresponds to at least 3 events (except outdoor activities which remain as one event)

## Event Constraints
1. **Time Continuity**: Events must be continuous time periods (current event end time matches next event start time), no time conflicts or gaps
2. **Outdoor Event Integrity**: Outdoor activities (work, shopping, walking, visiting friends, etc.) must be kept as a single event without decomposition. Room field should be "outdoor", time should cover the entire outdoor activity period
"""

EVENT_GENERATION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to decompose activities into events:

{event_requirements}

Based on the following resident profile, house layout, generated event context, and current batch activity list, decompose these activities into events.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Events (new events should connect with these)
{previous_events_context}

## Current Batch Activity List (to be decomposed)
{activities_batch}
"""

EVENT_VALIDATION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to validate event plans:

{event_requirements}

## Validation Focus
1. Time Verification: Check if events are continuous time periods (current event end time matches next event start time), no time conflicts or gaps should exist.
2. Logic Verification: Check if event decomposition is reasonable and granularity is appropriate, and if events can completely cover the original activity content.
3. Note: Only review the current event plan. Do not review resident profile, house layout, original activities, and context events.

Based on the following information, verify the rationality of the event plan.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Events (confirmed, for reference)
{previous_events_context}

## Original Activity Batch (to be decomposed)
{activities_batch}

## Current Event Plan (to be validated)
{events_json}
"""

EVENT_CORRECTION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to correct event plans:

{event_requirements}

Based on the following information, regenerate a corrected event plan. The corrected plan should connect seamlessly with the context.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Events (confirmed, do not modify, new events should connect with these)
{previous_events_context}

## Original Activity Batch (to be decomposed)
{activities_batch}

## Original Event Plan (to be corrected)
{original_events_json}

## Validation Issue Description
{correction_content}
"""

# ================================
# Action Step Agent Prompts
# ================================

ACTION_STEP_REQUIREMENTS = """
## Action Step Definition
Action steps are further refinements of events, divided into three distinct types:

### 1. Movement Type (movement)
- **Definition**: Process of person moving from one position to another, with obvious spatial displacement
- **Typical Scenarios**: Walking to wardrobe, moving to kitchen, walking from bed to desk, etc.
- **Coordinates**: Requires start_position and end_position (different positions)
- **Duration**: Usually short (several seconds to tens of seconds)

### 2. Stationary Type (stationary)
- **Definition**: Person conducts activities at a fixed position with no obvious displacement
- **Typical Scenarios**: Using furniture, operating items, resting, watching, eating, etc.
- **Coordinates**: Requires center_position and fluctuation_radius (usually ~0.1 meters)
- **Duration**: Flexible (tens of seconds to several hours)

### 3. Absent Type (absent)
- **Definition**: Person leaves home, not present in the house
- **Typical Scenarios**: Going to work, shopping, visiting friends, medical visits, etc.
- **Coordinates**: No coordinate information needed
- **Duration**: Flexible (can be long or short)

## Action Step Requirements
1. **Realistic Simulation**: Consider comprehensively, simulate real human behavior patterns, including natural movement and activity sequences. Each indoor event typically corresponds to at least 3 action steps((except outdoor events)
2. **Movement-Stationary Alternating Pattern**: Real behavior typically follows "movement → stationary → movement → stationary" pattern. Avoid:
   - Splitting one continuous movement into multiple consecutive movement actions
   - Missing movement actions between stationary actions at different locations
3. **Coordinate Accuracy**: 
   - Furniture position {x, y, width, height} gives bottom-left corner. Choose reasonable usage position:
     * Sleeping/resting on furniture: coordinates inside the furniture area
     * Operating furniture (table, stove): coordinates at external edge/operating area
   - Stationary fluctuation_radius usually around 0.1 meters
4. **Time Allocation**: Reasonably allocate time based on action complexity
5. **Style Consistency**: Maintain consistent granularity with previously generated action steps

## Action Step Constraints
1. **Type Separation**: Movement (displacement) and stationary (fixed position) cannot be merged into one action
2. **Time Continuity**: Adjacent action times must be seamless (previous end time = next start time)
3. **Space Continuity**: Adjacent action coordinates must connect reasonably
   - After movement: next action starts near movement end position
   - After stationary: next movement starts near stationary center position
   - After absent: next action must start at entrance door position
4. **Absent Decomposition**: Outdoor events must be decomposed into exactly 2 action steps:
   - Step 1 (movement): Move from current position to entrance door (end_position at entrance door)
   - Step 2 (absent): Leave home and return (no coordinates needed)
5. **Complete Coverage**: Action steps should completely reflect event content
"""

ACTION_STEP_GENERATION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to decompose events into action steps:

{action_step_requirements}

Based on the following resident profile, house layout, and events, decompose these events into action steps.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Action Steps (new actions should connect with these)
{previous_action_steps_context}

## Current Batch Event List (to be decomposed)
{events_batch}
"""

ACTION_STEP_VALIDATION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to validate action step plans:

{action_step_requirements}

## Validation Focus (only return false for serious errors)
1. Action Type Errors: Movement/stationary/absent types incorrectly classified or merged.
2. Coordinate Errors: Furniture usage coordinates obviously unreasonable (e.g., sleeping coordinates not in bed area).
3. Absent Type Errors: Absent action contains coordinates, or outdoor activity not properly decomposed (should be: movement to entrance door + absent action).
4. Entrance Door Connection: The movement step before absent must end at entrance door position; the action after absent must start at entrance door position.
5. Note: Only review the current action step plan. Return true when action decomposition logic is clear and types are correctly separated.

Based on the following information, verify the rationality of the action step plan.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Action Steps (confirmed, for reference)
{previous_action_steps_context}

## Original Event Batch (to be decomposed)
{events_batch}

## Current Action Step Plan (to be validated)
{action_steps_json}
"""

ACTION_STEP_CORRECTION_PROMPT_TEMPLATE = """
You are a professional behavior analyst, follow the following requirements to correct action step plans:

{action_step_requirements}

Based on the following information, regenerate a corrected action step plan. The corrected plan should connect seamlessly with the context.

## Resident Profile
{resident_profile}

## House Layout
{house_layout_yaml}

## Previously Generated Action Steps (confirmed, do not modify, new actions should connect with these)
{previous_action_steps_context}

## Original Event Batch (to be decomposed)
{events_batch}

## Original Action Step Plan (to be corrected)
{original_action_steps_json}

## Validation Issue Description
{correction_content}
"""

# ================================
# Prompt Template Objects
# ================================

def get_planning_prompts():
    """Get planning agent prompt templates"""
    return {
        'planning': ChatPromptTemplate.from_template(PLANNING_PROMPT_TEMPLATE),
        'validation': ChatPromptTemplate.from_template(PLANNING_VALIDATION_PROMPT_TEMPLATE),
        'correction': ChatPromptTemplate.from_template(PLANNING_CORRECTION_PROMPT_TEMPLATE)
    }

def get_event_prompts():
    """Get event agent prompt templates"""
    return {
        'generation': ChatPromptTemplate.from_template(EVENT_GENERATION_PROMPT_TEMPLATE),
        'validation': ChatPromptTemplate.from_template(EVENT_VALIDATION_PROMPT_TEMPLATE),
        'correction': ChatPromptTemplate.from_template(EVENT_CORRECTION_PROMPT_TEMPLATE)
    }

def get_action_step_prompts():
    """Get action step agent prompt templates"""
    return {
        'generation': ChatPromptTemplate.from_template(ACTION_STEP_GENERATION_PROMPT_TEMPLATE),
        'validation': ChatPromptTemplate.from_template(ACTION_STEP_VALIDATION_PROMPT_TEMPLATE),
        'correction': ChatPromptTemplate.from_template(ACTION_STEP_CORRECTION_PROMPT_TEMPLATE)
    }
