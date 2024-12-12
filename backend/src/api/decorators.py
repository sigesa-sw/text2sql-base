from functools import wraps
from flask import jsonify, request


def create_cache_decorator(app_instance):
    """Creates a requires_cache decorator with access to the app instance"""

    def requires_cache(required_fields, optional_fields=None):
        """
        Decorator to require cache fields
        Args:
            required_fields: List of required cache fields
            optional_fields: List of optional cache fields
        """
        if optional_fields is None:
            optional_fields = []

        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                cache_id = request.args.get("cache_id")
                if cache_id is None:
                    cache_id = request.json.get("cache_id")
                    if cache_id is None:
                        return jsonify({"type": "error", "error": "No id provided"})

                # Check required fields
                field_values = {}
                for field in required_fields:
                    value = app_instance.cache.get(cache_id=cache_id, field=field)
                    if value is None:
                        return jsonify({"type": "error", "error": f"No {field} found"})
                    field_values[field] = value

                # Get optional fields
                for field in optional_fields:
                    field_values[field] = app_instance.cache.get(
                        cache_id=cache_id, field=field
                    )

                # Add cache_id to field_values
                field_values["cache_id"] = cache_id

                return f(**field_values)

            return decorated

        return decorator

    return requires_cache
