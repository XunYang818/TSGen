YAML File Format Documentation
==================

```yaml

# Without style, suitable for large language models
house:
  size: { x: <house total width>, y: <house total height> }
  rooms:
    - name: <room name>
      position: { x: <bottom-left x>, y: <bottom-left y>,  width: <width>, height: <height> }
      walls:                # Wall settings (default draws top, right, bottom, left walls)
        top: true           # Or: { points: [[x1, y1], [x2, y2]] }
        right: true         # Or: { points: [[x1, y1], [x2, y2]] }
        bottom: true        # Or: { points: [[x1, y1], [x2, y2]] }
        left: true          # Or: { points: [[x1, y1], [x2, y2]] }
        extra:              # Optional, additional wall list
          - points: [[x1, y1], [x2, y2]]
      doors:                     # Door list
        - name: <door name>
          points: [[x1, y1], [x2, y2], [x3, y3]]
      furniture:                 # Furniture list
        - name: <furniture name>
          id: <furniture id>
          position: { x: <bottom-left x>, y: <bottom-left y>, width: <width>, height: <height> }
```

```yaml

# Without style
house:
  size: { x: 100, y: 100 }
  rooms:
    - name: Bathroom
      position: { x: 0, y: 40, width: 30, height: 20 }
      walls:
        top: true
        right: true
        bottom: true
        left: true
      doors:
        - name: Bathroom Door
          points: [[30, 41], [30, 50], [20, 41]]
      furniture:
        - name: Toilet
          id: toilet_001
          position: { x: 1, y: 50, width: 8, height: 9 }
        - name: Sink
          id: sink_001
          position: { x: 15, y: 50, width: 14, height: 9 }

    - name: Kitchen
      position: { x: 20, y: 70, width: 20, height: 30 }
      walls:
        top: true
        right: true
        bottom: true
        left: true
      doors:
        - name: Kitchen Door
          points: [[30, 70], [39, 70], [39, 80]]
      furniture:
        - name: Refrigerator
          id: refrigerator_001
          position: { x: 21, y: 71, width: 7, height: 7 }
        - name: Stove
          id: stove_001
          position: { x: 21, y: 80, width: 7, height: 19 }
        - name: Sink
          id: sink_002
          position: { x: 30, y: 90, width: 9, height: 9 }

    - name: Secondary Bedroom
      position: { x: 40, y: 70, width: 30, height: 30 }
      walls:
        top: true
        right: true
        bottom: true
        left: true
      doors:
        - name: Secondary Bedroom Door
          points: [[41, 70], [50, 70], [41, 80]]
      furniture:
        - name: Bed
          id: bed_001
          position: { x: 50, y: 81, width: 19, height: 15 }
        - name: Wardrobe
          id: wardrobe_001
          position: { x: 41, y: 80, width: 5, height: 19 }
        - name: Desk
          id: desk_001
          position: { x: 61, y: 71, width: 8, height: 9 }

    - name: Master Bedroom
      position: { x: 45, y: 0, width: 25, height: 40 }
      walls:
        top: true
        right: true
        bottom: true
        left: true
      doors:
        - name: Master Bedroom Door
          points: [[46, 40], [55, 40], [46, 30]]
      furniture:
        - name: Bed
          id: bed_002
          position: { x: 50, y: 10, width: 19, height: 18 }
        - name: Wardrobe
          id: wardrobe_002
          position: { x: 54, y: 34, width: 15, height: 5 }

    - name: Living Room 1
      position: { x: 30, y: 40, width: 40, height: 30 }
      walls:
        top: true
        right: true
        bottom: true
        left: true
      doors:
        - name: Entrance Door
          points: [[30, 61], [30, 69], [40, 69]]
      furniture:
        - name: Dining Table
          id: dining_table_001
          position: { x: 47.5, y: 49, width: 15, height: 12 }

    - name: Living Room 2
      position: { x: 10, y: 0, width: 35, height: 40 }
      walls:
        top: true
        right: false
        bottom: true
        left: true
      doors:
        - name: Living Room Door
          points: [[44, 40], [35, 40], [44, 30]]
      furniture:
        - name: Sofa
          id: sofa_001
          position: { x: 11, y: 10, width: 8, height: 20 }
        - name: Coffee Table
          id: coffee_table_001
          position: { x: 25, y: 14, width: 5, height: 12 }
        - name: TV Cabinet
          id: tv_cabinet_001
          position: { x: 40, y: 11, width: 4, height: 18 }
        - name: Treadmill
          id: treadmill_001
          position: { x: 29, y: 1, width: 15, height: 8 }
```

```yaml

# With style, suitable for drawing
house:
  size: { x: <house total width>, y: <house total height> }
  resolution: <resolution>       # Optional, default value 100
  rooms:
    - name: <room name>
      position: { x: <bottom-left x>, y: <bottom-left y>,  width: <width>, height: <height> }
      walls:                # Wall settings (default draws top, right, bottom, left walls)
        top: true           # Or: { points: [[x1, y1], [x2, y2]] }
        right: true         # Or: { points: [[x1, y1], [x2, y2]] }
        bottom: true        # Or: { points: [[x1, y1], [x2, y2]] }
        left: true          # Or: { points: [[x1, y1], [x2, y2]] }
        extra:              # Optional, additional wall list
          - points: [[x1, y1], [x2, y2]]
            linewidth: <line width>     # Default 5
            color: <color>              # Default "gray"
      floor:                     # Floor settings
        color: <color>             # Default "#F6E9CA"
        hatch: <fill pattern>      # '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*' 
      doors:                     # Door list
        - name: <door name>
          points: [[x1, y1], [x2, y2], [x3, y3]]
          width: <door width>      # Default 6
          color: <color>           # Default "white"
          alpha: <opacity>         # Default 0.5
          padding: <inner padding> # Default 0
      furniture:                 # Furniture list
        - name: <furniture name>
          id: <furniture id>       # e.g. livingroom-bed
          type: <type>             # "rectangle" or "image", default "rectangle"
          position: { x: <bottom-left x>, y: <bottom-left y>, width: <width>, height: <height> }
          rotation: <rotation angle>    # Default 0
          padding: <inner padding>      # Single value (same for all sides)
          config:                       # Configuration for rectangular furniture
            color: <color>              # Default "brown"
            fontsize: <font size>       # Default 8
            hatch: <fill pattern>       # Optional
          image: <image path>           # Used when type is "image"
```