import json

try:
    import orjson
except ImportError:
    orjson = None


def _json_dumps(obj):
    if orjson is not None:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode("utf-8")
    return json.dumps(obj, indent=2)
