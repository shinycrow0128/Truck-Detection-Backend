from datetime import datetime

from supabase import Client


def insert_video_row(supabase: Client, dt: datetime, video_url: str):
    try:
        response = (
            supabase.table("video")
            .insert({"date": dt.isoformat(), "video_url": video_url})
            .execute()
        )

        if response.data and len(response.data) > 0:
            inserted_row = response.data[0]
            return inserted_row["id"]
        print("[Supabase] Insert appeared to succeed but no row returned")
        return None
    except Exception as e:
        print(f"[Supabase] Failed: {e}")
        return None


def get_camera_id(supabase: Client, camera_name: str) -> str | None:
    """Get camera id by camera_name."""
    try:
        response = (
            supabase.table("camera")
            .select("id")
            .eq("camera_name", camera_name)
            .maybe_single()
            .execute()
        )

        if response.data:
            return response.data["id"]
        return None

    except Exception as e:
        print(f"Error while fetching camera '{camera_name}': {e}")
        return None


def get_truck_id(supabase: Client, truck_name: int | str) -> str | None:
    """Get truck id by truck_name."""
    try:
        search_name = f"Truck-{int(truck_name) + 1}"
        response = (
            supabase.table("truck")
            .select("id")
            .eq("truck_name", search_name)
            .maybe_single()
            .execute()
        )

        if response.data:
            return response.data["id"]
        return None

    except Exception as e:
        print(f"Error while fetching truck '{truck_name}': {e}")
        return None
