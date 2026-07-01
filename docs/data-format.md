# Trajectory Simulation System JSON File Format Specification

## Overview

The trajectory simulation system generates a complete JSON file, saved in the format `{yaml_number}_{occupation}.json`. This file contains the complete data hierarchy from resident profile analysis to final trajectory generation.

## File Structure Overview

```json
{
  "info": {},                     // Statistics info (LLM usage, counts)
  "simulation_id": "string",      // Unique simulation ID
  "activity_plans": [],           // Activity planning list
  "events": [],                   // Events list
  "action_steps": [],             // Action steps list
  "trajectories": [],             // Trajectories list
  "created_at": "timestamp"       // Creation timestamp
}
```

### Info (Statistics)

```json
{
  "counts": {
    "plans": 15,
    "events": 45,
    "actions": 120
  },
  "usage": {
    "agents": {
      "plan_gen": {"calls": 1, "input_tokens": 5000, "output_tokens": 2000, "total_tokens": 7000},
      "plan_val": {"calls": 1, "input_tokens": 6000, "output_tokens": 500, "total_tokens": 6500},
      "plan_cor": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
      "event_gen": {"calls": 15, "...": "..."},
      "event_val": {"calls": 15, "...": "..."},
      "event_cor": {"calls": 2, "...": "..."},
      "action_gen": {"calls": 45, "...": "..."},
      "action_val": {"calls": 45, "...": "..."},
      "action_cor": {"calls": 5, "...": "..."}
    },
    "pass_once": {
      "plan": 1,
      "event": 28,
      "action": 40
    },
    "totals": {
      "total_calls": 129,
      "total_input_tokens": 280000,
      "total_output_tokens": 55000,
      "total_tokens": 335000
    }
  }
}
```

## Detailed Field Descriptions

### 1. Activity Plans (activity_plans)

Each activity plan contains the following fields:

```json
{
  "activity_id": "1",                              // Unique incremental ID of activity (e.g., "1", "2", "3", ...)
  "activity_name": "Wake up and wash",             // Activity name
  "start_time": "2025-01-01 07:00:00",            // Start time (format: YYYY-MM-DD HH:MM:SS)
  "end_time": "2025-01-01 07:18:00",              // End time (format: YYYY-MM-DD HH:MM:SS)
  "rooms": ["Master Bedroom", "Bathroom"],         // List of main rooms involved (empty if outdoor)
  "description": "Get up from bed, do simple stretching exercises, then go to bathroom to wash.",
  "purpose": "Start a new day, maintain personal hygiene.",
  "related_objects": [                             // List of related objects
    "Bed_002",
    "Washbasin_001"
  ],
  "activity_type": "Personal Affairs"              // Activity type
}
```

### 2. Events (events)

Each event contains the following fields:

```json
{
  "event_id": "1",                                 // Unique incremental ID (e.g., "1", "2", "3", ...)
  "activity_id": "1",                              // Parent activity ID
  "event_name": "Wake up",                         // Event name
  "start_time": "2025-01-01 07:00:00",            // Start time (format: YYYY-MM-DD HH:MM:SS)
  "end_time": "2025-01-01 07:02:00",              // End time (format: YYYY-MM-DD HH:MM:SS)
  "room": "Master Bedroom",                        // Room where event occurs
  "description": "Get up from bed, do simple stretching exercises to wake up the body.",
  "purpose": "Wake up the body, prepare to start daily activities.",
  "related_objects": [                             // Related objects
    "Bed_002"
  ],
  "event_type": "Personal Preparation"             // Event type
}
```

### 3. Action Steps (action_steps)

Each action step contains the following fields:

```json
{
  "action_step_id": "1",                           // Unique incremental ID (e.g., "1", "2", "3", ...)
  "event_id": "1",                                 // Parent event ID
  "action_step_name": "Get up from bed",           // Action step name
  "start_time": "2025-01-01 07:00:00",            // Start time (format: YYYY-MM-DD HH:MM:SS)
  "end_time": "2025-01-01 07:00:02",              // End time (format: YYYY-MM-DD HH:MM:SS)
  "action_type": "movement",                       // Action type (movement/stationary/absent)
  "room": "Master Bedroom",                        // Room where action occurs
  "description": "Move from bed center to bed edge.",

  // For movement type:
  "movement_detail": {
    "path_description": "Moving from bed center to bed edge, preparing for stretching exercises.",
    "start_position": {
      "x": 59.5,                                   // X coordinate (decimeters)
      "y": 19.0                                    // Y coordinate (decimeters)
    },
    "end_position": {
      "x": 59.5,                                   // X coordinate (decimeters)
      "y": 28.0                                    // Y coordinate (decimeters)
    }
  },
  "stationary_detail": null,                       // Null for non-stationary types
  "absent_detail": null                            // Null for non-absent types
}
```

#### Stationary Type Detail Fields

```json
"stationary_detail": {
  "activity_description": "Operating rice cooker in kitchen",
  "center_position": {
    "x": 40.0,
    "y": 20.0
  },
  "fluctuation_radius": 0.3                        // Fluctuation radius (meters)
}
```

#### Absent Type Detail Fields

```json
"absent_detail": {
  "reason": "Going out shopping",
  "location_description": "Nearby supermarket"
}
```

### 4. Trajectories (trajectories)

Each trajectory contains the following fields:

```json
{
  "trajectory_id": "1",                            // Unique incremental ID (e.g., "1", "2", "3", ...)
  "action_step_id": "1",                           // Corresponding action step ID
  "points": [                                      // List of trajectory points
    {
      "x": 59.0,                                   // X coordinate (decimeters)
      "y": 19.0,                                   // Y coordinate (decimeters)
      "timestamp": "2025-01-01 07:00:00"           // Timestamp (format: YYYY-MM-DD HH:MM:SS)
    },
    {
      "x": 59.0,
      "y": 28.0,
      "timestamp": "2025-01-01 07:00:02"
    }
  ]
}
```

## Coordinate System

- **Unit**: Decimeters (dm)
- **Origin**: Bottom-left corner of the house
- **X-axis**: Left to right (positive)
- **Y-axis**: Bottom to top (positive)

## Sampling Rate

- **Movement trajectories**: 0.5 seconds per point (default)
- **Stationary trajectories**: 0.5 seconds per point (default)
