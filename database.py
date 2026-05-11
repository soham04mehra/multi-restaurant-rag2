# REMINDER: The .env file contains sensitive keys and must NEVER be committed to version control!
from supabase import create_client, Client
import config

# Initialize the client once globally for the whole project
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

def insert_dish(row_data: dict):
    """Inserts a dish using the global supabase connection."""
    return supabase.table("menu_items").insert(row_data).execute()

def delete_all_dishes(restaurant_id: str):
    """Deletes all dishes for a specific restaurant to allow clean re-ingestion."""
    return supabase.table("menu_items").delete().eq("restaurant_id", restaurant_id).execute()
