"""Schema adapter for grocery_delivery.db.

This module:
1. Discovers all tables and columns in your database
2. Helps you map columns from ANY table to the expected schema
3. Automatically figures out how to JOIN tables to get all needed columns
4. Creates a view that the ML pipeline can use

Run this first to understand your database structure:
    python -m delivery_ml.data.schema_adapter
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from delivery_ml.config import settings


def discover_schema(db_path: str | None = None) -> dict[str, Any]:
    """
    Discover the schema of your grocery_delivery.db.
    
    Returns detailed information about all tables, columns, and foreign keys.
    """
    db_path = db_path or str(settings.sqlite_db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all tables
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = [row[0] for row in cursor.fetchall()]
    
    result = {"database": db_path, "tables": {}, "all_columns": {}}
    
    for table_name in table_names:
        # Get column info
        col_cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in col_cursor.fetchall():
            col_info = {
                "name": row[1],
                "type": row[2],
                "nullable": not row[3],
                "primary_key": bool(row[5]),
            }
            columns.append(col_info)
            
            # Track all columns globally for easy lookup
            full_col_name = f"{table_name}.{row[1]}"
            result["all_columns"][full_col_name] = {
                "table": table_name,
                "column": row[1],
                "type": row[2],
            }
        
        # Get foreign keys
        fk_cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = [
            {
                "from_column": row[3],
                "to_table": row[2],
                "to_column": row[4],
            }
            for row in fk_cursor.fetchall()
        ]
        
        # Get row count
        count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = count_cursor.fetchone()[0]
        
        # Get sample data
        sample_cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample = [dict(row) for row in sample_cursor.fetchall()]
        
        result["tables"][table_name] = {
            "columns": columns,
            "foreign_keys": foreign_keys,
            "row_count": row_count,
            "sample_data": sample,
        }
    
    # Infer relationships even without explicit foreign keys
    result["inferred_relationships"] = _infer_relationships(result["tables"])
    
    conn.close()
    return result


def _infer_relationships(tables: dict) -> list[dict]:
    """
    Infer relationships between tables based on column naming conventions.
    
    Looks for patterns like:
    - table_name.id -> other_table.table_name_id
    - orders.customer_id -> customers.id
    """
    relationships = []
    
    # Build a map of potential primary keys
    pk_candidates = {}
    for table_name, table_info in tables.items():
        for col in table_info["columns"]:
            if col["primary_key"] or col["name"] == "id":
                pk_candidates[table_name] = col["name"]
    
    # Look for foreign key patterns
    for table_name, table_info in tables.items():
        for col in table_info["columns"]:
            col_name = col["name"].lower()
            
            # Pattern: xxx_id might reference xxx table
            if col_name.endswith("_id") and col_name != "id":
                referenced_table = col_name[:-3]  # Remove _id
                
                # Check if that table exists
                for other_table in tables.keys():
                    if other_table.lower() == referenced_table or \
                       other_table.lower() == referenced_table + "s" or \
                       other_table.lower().rstrip("s") == referenced_table:
                        
                        # Find the PK of the referenced table
                        ref_pk = pk_candidates.get(other_table, "id")
                        
                        relationships.append({
                            "from_table": table_name,
                            "from_column": col["name"],
                            "to_table": other_table,
                            "to_column": ref_pk,
                            "inferred": True,
                        })
    
    return relationships


def print_schema_report(schema: dict[str, Any]) -> None:
    """Print a human-readable schema report."""
    print(f"\n{'='*60}")
    print(f"DATABASE: {schema['database']}")
    print(f"{'='*60}\n")
    
    for table_name, table_info in schema["tables"].items():
        print(f"\n📋 TABLE: {table_name}")
        print(f"   Rows: {table_info['row_count']:,}")
        print(f"   Columns:")
        
        for col in table_info["columns"]:
            pk = " [PK]" if col["primary_key"] else ""
            nullable = " (nullable)" if col["nullable"] else ""
            print(f"      - {col['name']}: {col['type']}{pk}{nullable}")
        
        if table_info["foreign_keys"]:
            print(f"   Foreign Keys:")
            for fk in table_info["foreign_keys"]:
                print(f"      - {fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")
        
        if table_info["sample_data"]:
            print(f"\n   Sample row:")
            for key, value in list(table_info["sample_data"][0].items())[:8]:
                print(f"      {key}: {value}")
            if len(table_info["sample_data"][0]) > 8:
                print(f"      ... and {len(table_info['sample_data'][0]) - 8} more columns")
    
    if schema.get("inferred_relationships"):
        print(f"\n{'='*60}")
        print("INFERRED RELATIONSHIPS")
        print(f"{'='*60}")
        for rel in schema["inferred_relationships"]:
            print(f"   {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")


# Expected schema for the ML pipeline
EXPECTED_COLUMNS = {
    "order_id": "Unique order identifier",
    "customer_id": "Customer identifier",
    "store_id": "Restaurant/store identifier", 
    "created_at": "When order was placed (datetime)",
    "delivered_at": "When order was delivered (datetime, nullable)",
    "latitude": "Pickup location latitude",
    "longitude": "Pickup location longitude",
    "delivery_latitude": "Dropoff location latitude",
    "delivery_longitude": "Dropoff location longitude",
    "total": "Order value in cents",
    "quantity": "Number of items",
}


def generate_mapping_template(schema: dict[str, Any], output_path: str = "column_mapping.json") -> None:
    """
    Generate a JSON template for column mapping.
    
    Lists ALL available columns from ALL tables so you can pick which ones to use.
    """
    template = {
        "_instructions": [
            "Map your columns to the expected ML pipeline columns below.",
            "Use 'table.column' format to specify columns from any table.",
            "The system will automatically figure out how to JOIN the tables.",
            "delivery_time_minutes is CALCULATED from (delivered_at - created_at), don't map it.",
        ],
        "column_mapping": {},
        "available_columns": {},
    }
    
    # List all available columns by table
    for table_name, table_info in schema["tables"].items():
        template["available_columns"][table_name] = [
            col["name"] for col in table_info["columns"]
        ]
    
    # Pre-fill with best guesses
    for expected_col, description in EXPECTED_COLUMNS.items():
        template["column_mapping"][expected_col] = {
            "maps_to": _guess_column_mapping(expected_col, schema),
            "description": description,
        }
    
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✅ Generated mapping template: {output_path}")
    print("\nEdit the 'maps_to' values to point to your actual columns.")
    print("Use format: 'table_name.column_name'")
    print("\nExample:")
    print('  "order_id": {"maps_to": "orders.id", ...}')
    print('  "latitude": {"maps_to": "stores.latitude", ...}')


def _guess_column_mapping(expected_col: str, schema: dict) -> str:
    """Try to guess which column maps to the expected column."""
    expected_lower = expected_col.lower()
    
    # Common mappings to try
    variations = {
        "order_id": ["order_id", "id", "order_number"],
        "customer_id": ["customer_id", "user_id", "buyer_id"],
        "store_id": ["restaurant_id", "store_id", "vendor_id", "merchant_id"],
        "created_at": ["created_at", "placed_at", "order_date", "order_time", "timestamp"],
        "delivered_at": ["delivered_at", "delivery_date", "completed_at"],
        "latitude": ["restaurant_latitude", "store_latitude", "pickup_lat", "latitude", "lat"],
        "longitude": ["restaurant_longitude", "store_longitude", "pickup_lng", "longitude", "lng", "lon"],
        "delivery_latitude": ["delivery_latitude", "dropoff_lat", "customer_latitude", "dest_lat"],
        "delivery_longitude": ["delivery_longitude", "dropoff_lng", "customer_longitude", "dest_lng", "dest_lon"],
        "total": ["order_total_cents", "total_cents", "amount", "total", "price", "order_value"],
        "quantity": ["item_count", "num_items", "quantity", "items_count", "total_items"],
    }
    
    search_terms = variations.get(expected_lower, [expected_lower])
    
    # Search all tables for matching columns
    for table_name, table_info in schema["tables"].items():
        for col in table_info["columns"]:
            col_lower = col["name"].lower()
            for term in search_terms:
                if term in col_lower or col_lower in term:
                    return f"{table_name}.{col['name']}"
    
    return "YOUR_TABLE.YOUR_COLUMN"


def build_join_query(
    schema: dict[str, Any],
    column_mapping: dict[str, str],
    base_table: str | None = None,
) -> str:
    """
    Build a SELECT query with JOINs based on the column mapping.
    
    Automatically figures out how to join tables based on:
    1. Explicit foreign keys
    2. Inferred relationships (xxx_id -> xxx.id patterns)
    """
    # Parse which tables we need
    tables_needed = set()
    select_parts = []
    
    for expected_col, mapping in column_mapping.items():
        if isinstance(mapping, dict):
            source = mapping.get("maps_to", "")
        else:
            source = mapping
            
        if "." in source and source != "YOUR_TABLE.YOUR_COLUMN":
            table, col = source.split(".", 1)
            tables_needed.add(table)
            select_parts.append(f"{table}.{col} AS {expected_col}")
    
    if not tables_needed:
        raise ValueError("No valid column mappings found")
    
    # Determine base table (the one with the most columns, likely orders)
    if base_table is None:
        table_col_counts = {}
        for expected_col, mapping in column_mapping.items():
            if isinstance(mapping, dict):
                source = mapping.get("maps_to", "")
            else:
                source = mapping
            if "." in source:
                table = source.split(".")[0]
                table_col_counts[table] = table_col_counts.get(table, 0) + 1
        
        base_table = max(table_col_counts.keys(), key=lambda t: table_col_counts[t])
    
    tables_needed.discard(base_table)
    
    # Build relationships map
    relationships = {}
    
    # From explicit foreign keys
    for table_name, table_info in schema["tables"].items():
        for fk in table_info.get("foreign_keys", []):
            key = (table_name, fk["to_table"])
            relationships[key] = (fk["from_column"], fk["to_column"])
            # Also store reverse
            key_rev = (fk["to_table"], table_name)
            relationships[key_rev] = (fk["to_column"], fk["from_column"])
    
    # From inferred relationships
    for rel in schema.get("inferred_relationships", []):
        key = (rel["from_table"], rel["to_table"])
        relationships[key] = (rel["from_column"], rel["to_column"])
        key_rev = (rel["to_table"], rel["from_table"])
        relationships[key_rev] = (rel["to_column"], rel["from_column"])
    
    # Build JOIN clauses
    join_clauses = []
    joined_tables = {base_table}
    
    for table in tables_needed:
        join_found = False
        
        # Try direct join to base table
        key = (base_table, table)
        if key in relationships:
            base_col, other_col = relationships[key]
            join_clauses.append(
                f"LEFT JOIN {table} ON {base_table}.{base_col} = {table}.{other_col}"
            )
            joined_tables.add(table)
            join_found = True
        else:
            # Try reverse
            key = (table, base_table)
            if key in relationships:
                other_col, base_col = relationships[key]
                join_clauses.append(
                    f"LEFT JOIN {table} ON {base_table}.{base_col} = {table}.{other_col}"
                )
                joined_tables.add(table)
                join_found = True
        
        if not join_found:
            # Try joining through an intermediate table
            for intermediate in joined_tables.copy():
                key = (intermediate, table)
                if key in relationships:
                    int_col, other_col = relationships[key]
                    join_clauses.append(
                        f"LEFT JOIN {table} ON {intermediate}.{int_col} = {table}.{other_col}"
                    )
                    joined_tables.add(table)
                    join_found = True
                    break
                
                key = (table, intermediate)
                if key in relationships:
                    other_col, int_col = relationships[key]
                    join_clauses.append(
                        f"LEFT JOIN {table} ON {intermediate}.{int_col} = {table}.{other_col}"
                    )
                    joined_tables.add(table)
                    join_found = True
                    break
        
        if not join_found:
            print(f"⚠️  Warning: Could not find how to join table '{table}'")
            print(f"   You may need to add a manual JOIN condition")
    
    # Build final query
    select_clause = ",\n    ".join(select_parts)
    join_clause = "\n".join(join_clauses)
    
    query = f"""SELECT
    {select_clause}
FROM {base_table}
{join_clause}
WHERE {base_table}.delivered_at IS NOT NULL"""
    
    return query, base_table


def create_orders_view(
    db_path: str,
    column_mapping: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> None:
    """
    Create a view that joins tables and maps columns to expected schema.
    """
    if schema is None:
        schema = discover_schema(db_path)
    
    # Clean up mapping (handle nested dict format)
    clean_mapping = {}
    for expected_col, mapping in column_mapping.items():
        if isinstance(mapping, dict):
            clean_mapping[expected_col] = mapping.get("maps_to", "")
        else:
            clean_mapping[expected_col] = mapping
    
    query, base_table = build_join_query(schema, clean_mapping)
    
    print("\n📝 Generated query:")
    print("-" * 40)
    print(query)
    print("-" * 40)
    
    conn = sqlite3.connect(db_path)
    
    # Drop existing view if exists
    conn.execute("DROP VIEW IF EXISTS orders_ml_view")
    
    # Create view
    view_sql = f"CREATE VIEW orders_ml_view AS\n{query}"
    
    try:
        conn.execute(view_sql)
        conn.commit()
        print("\n✅ Created view 'orders_ml_view'")
        
        # Test it
        cursor = conn.execute("SELECT COUNT(*) FROM orders_ml_view")
        count = cursor.fetchone()[0]
        print(f"   View contains {count:,} rows")
        
        # Show sample
        cursor = conn.execute("SELECT * FROM orders_ml_view LIMIT 1")
        row = cursor.fetchone()
        if row:
            print("\n   Sample row:")
            for i, col in enumerate(cursor.description):
                print(f"      {col[0]}: {row[i]}")
                
    except sqlite3.Error as e:
        print(f"\n❌ Error creating view: {e}")
        print("\nYou may need to manually adjust the JOIN conditions.")
    
    conn.close()


def update_config_for_view() -> None:
    """
    Update the feature computation to use orders_ml_view instead of orders.
    """
    print("\n📌 Next step: Update your queries to use 'orders_ml_view' instead of 'orders'")
    print("   Or rename the view to 'orders' if you prefer")


if __name__ == "__main__":
    import sys
    
    db_path = str(settings.sqlite_db_path)
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Make sure grocery_delivery.db is in the current directory")
        print("Or set DELIVERY_ML_SQLITE_DB_PATH environment variable")
        sys.exit(1)
    
    # Discover and print schema
    print("Discovering schema...")
    schema = discover_schema(db_path)
    print_schema_report(schema)
    
    # Handle command line args
    if "--generate-template" in sys.argv:
        generate_mapping_template(schema)
        sys.exit(0)
    
    if "--apply-mapping" in sys.argv:
        idx = sys.argv.index("--apply-mapping")
        if idx + 1 < len(sys.argv):
            mapping_file = sys.argv[idx + 1]
            with open(mapping_file) as f:
                mapping_config = json.load(f)
            
            create_orders_view(
                db_path,
                mapping_config["column_mapping"],
                schema,
            )
            sys.exit(0)
        else:
            print("Error: --apply-mapping requires a file path")
            sys.exit(1)
    
    if "--test-query" in sys.argv:
        idx = sys.argv.index("--test-query")
        if idx + 1 < len(sys.argv):
            mapping_file = sys.argv[idx + 1]
            with open(mapping_file) as f:
                mapping_config = json.load(f)
            
            # Clean up mapping
            clean_mapping = {}
            for expected_col, mapping in mapping_config["column_mapping"].items():
                if isinstance(mapping, dict):
                    clean_mapping[expected_col] = mapping.get("maps_to", "")
                else:
                    clean_mapping[expected_col] = mapping
            
            query, base_table = build_join_query(schema, clean_mapping)
            print("\n📝 Generated query:")
            print("-" * 40)
            print(query)
            print("-" * 40)
            sys.exit(0)
    
    # Default: show instructions
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("\n1. Generate a mapping template:")
    print("   python -m delivery_ml.data.schema_adapter --generate-template")
    print("\n2. Edit column_mapping.json to map your columns")
    print("   Use 'table.column' format, e.g., 'stores.latitude'")
    print("\n3. Test the generated query:")
    print("   python -m delivery_ml.data.schema_adapter --test-query column_mapping.json")
    print("\n4. Create the view:")
    print("   python -m delivery_ml.data.schema_adapter --apply-mapping column_mapping.json")
