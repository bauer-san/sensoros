# zone_assignment.py
from shapely.geometry import Point, Polygon
import json

class ZoneAssigner:
    def __init__(self, zone_config_file="zone_config.json"):
        with open(zone_config_file) as f:
            data = json.load(f)

        self.zones = [
            {
                "id": z["id"],
                "label": z["label"],
                "alert_level": z["alert_level"],
                "polygon": Polygon(z["polygon"])
            }
            for z in data["zones"]
        ]

        # Sort by area ascending — assign to smallest
        # containing zone (most specific)
        self.zones.sort(key=lambda z: z["polygon"].area)

    def assign(self, world_x: float, world_y: float) -> dict | None:
        """
        Returns the most specific zone containing this point,
        or None if outside all defined zones.
        """
        pt = Point(world_x, world_y)
        for zone in self.zones:
            if zone["polygon"].contains(pt):
                return {
                    "id": zone["id"],
                    "label": zone["label"],
                    "alert_level": zone["alert_level"]
                }
        return None