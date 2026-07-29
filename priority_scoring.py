"""
Priority Scoring Module
========================
Combines: severity + nearby schools/hospitals (real, via Google Places API)
          + traffic density + waterlogging risk (seeded/dummy for demo,
          since free real-time APIs for these aren't readily available)

Usage:
    python priority_scoring.py --lat 28.6139 --lon 77.2090 --severity 6.4

Requires:
    pip install requests
    Set your Google Places API key below or as env var GOOGLE_PLACES_API_KEY
"""
import os
import argparse
import requests

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "YOUR_API_KEY_HERE")
NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Search radius for schools/hospitals (meters) - a pothole within this
# distance of a school/hospital is considered higher-risk
PROXIMITY_RADIUS_M = 500


def get_nearby_place_count(lat, lon, place_type, radius=PROXIMITY_RADIUS_M):
    """
    Query Google Places API (Nearby Search) for the count of a given place
    type within radius meters of the given coordinates.
    place_type examples: 'school', 'hospital'
    """
    if GOOGLE_PLACES_API_KEY == "YOUR_API_KEY_HERE":
        print("WARNING: No Google Places API key set - skipping real lookup, returning 0")
        return 0

    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": place_type,
        "key": GOOGLE_PLACES_API_KEY,
    }
    try:
        resp = requests.get(NEARBY_SEARCH_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            print(f"WARNING: Places API returned status={data.get('status')} for type={place_type}")
            return 0
        return len(data.get("results", []))
    except requests.RequestException as e:
        print(f"WARNING: Places API request failed: {e}")
        return 0


def get_traffic_density_score(lat, lon):
    """
    PLACEHOLDER: real-time traffic density needs a paid API (Google Roads /
    Distance Matrix with traffic model, or a live traffic data provider).
    For the hackathon demo, this is seeded/dummy - returns a fixed mid-range
    value. Swap this out for a real API call if you get traffic data access
    (e.g. TomTom Traffic API has a free tier you could try instead).
    """
    # TODO: replace with real traffic API call when available
    SEEDED_TRAFFIC_SCORE = 5.0  # out of 10, mid-range placeholder
    return SEEDED_TRAFFIC_SCORE


def get_waterlogging_risk_score(lat, lon):
    """
    PLACEHOLDER: no simple free live waterlogging API exists. For the demo,
    seed this from a small manually-curated list of known flood-prone areas
    (e.g. from local municipal reports or news), else default to a low score.
    Replace SEEDED_WATERLOGGING_ZONES with real known problem spots in your
    demo city before presenting.
    """
    SEEDED_WATERLOGGING_ZONES = [
        # (lat, lon, radius_km, risk_score_out_of_10)
        # Example entries - REPLACE with real known waterlogging spots for your demo city
        (28.6139, 77.2090, 2.0, 8.0),   # example: central Delhi flood-prone zone
    ]

    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    for zone_lat, zone_lon, radius_km, risk in SEEDED_WATERLOGGING_ZONES:
        if haversine_km(lat, lon, zone_lat, zone_lon) <= radius_km:
            return risk

    return 1.0  # default low risk if not in any known zone


def compute_priority_score(severity, lat, lon):
    """
    Combines all factors into a final priority score (0-10).
    Weights are a starting point for the hackathon demo - tune these if you
    get real data and want to rebalance factor importance.
    """
    school_count = get_nearby_place_count(lat, lon, "school")
    hospital_count = get_nearby_place_count(lat, lon, "hospital")

    # Cap counts so one area with 20 schools doesn't dominate the score
    proximity_score = min(10, (school_count + hospital_count) * 2)

    traffic_score = get_traffic_density_score(lat, lon)
    waterlogging_score = get_waterlogging_risk_score(lat, lon)

    WEIGHTS = {
        "severity": 0.40,
        "proximity": 0.25,   # schools + hospitals nearby
        "traffic": 0.20,
        "waterlogging": 0.15,
    }

    final_score = (
        severity * WEIGHTS["severity"]
        + proximity_score * WEIGHTS["proximity"]
        + traffic_score * WEIGHTS["traffic"]
        + waterlogging_score * WEIGHTS["waterlogging"]
    )

    return {
        "priority_score": round(min(10, final_score), 1),
        "breakdown": {
            "severity": severity,
            "school_count": school_count,
            "hospital_count": hospital_count,
            "proximity_score": proximity_score,
            "traffic_score": traffic_score,
            "waterlogging_score": waterlogging_score,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True, help="Latitude of pothole")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of pothole")
    parser.add_argument("--severity", type=float, required=True, help="Severity score (0-10) from detection model")
    args = parser.parse_args()

    result = compute_priority_score(args.severity, args.lat, args.lon)

    print(f"\nPriority Score: {result['priority_score']}/10")
    print("Breakdown:")
    for k, v in result["breakdown"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
