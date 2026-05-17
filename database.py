from supabase import create_async_client, AsyncClient
import config

_supabase_client = None

async def get_supabase() -> AsyncClient:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = await create_async_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase_client

async def insert_dish(row_data: dict):
    """Inserts a dish using the global async supabase connection."""
    client = await get_supabase()
    return await client.table("menu_items").insert(row_data).execute()

async def delete_all_dishes(restaurant_id: str):
    """Deletes all dishes for a specific restaurant asynchronously."""
    client = await get_supabase()
    return await client.table("menu_items").delete().eq("restaurant_id", restaurant_id).execute()
