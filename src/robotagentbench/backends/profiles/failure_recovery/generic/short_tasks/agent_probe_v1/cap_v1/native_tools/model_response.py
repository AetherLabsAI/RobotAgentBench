"""Validate a whole model response before dispatching any robot action."""
import json
import math


class ModelFormatError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelFormatError('duplicate argument field')
        result[key] = value
    return result


def _validate(value, schema):
    kind = schema['type']
    if kind == 'object':
        if not isinstance(value, dict):
            raise ModelFormatError('expected object')
        props = schema.get('properties', {})
        if set(schema.get('required', [])) - value.keys():
            raise ModelFormatError('missing required argument')
        if schema.get('additionalProperties') is False and value.keys() - props.keys():
            raise ModelFormatError('unknown argument field')
        for key, item in value.items():
            _validate(item, props[key])
    elif kind == 'number':
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ModelFormatError('expected finite number')
    elif kind == 'string':
        if not isinstance(value, str):
            raise ModelFormatError('expected string')
    elif kind == 'array':
        if not isinstance(value, list) or not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')):
            raise ModelFormatError('invalid array length/type')
        for item in value:
            _validate(item, schema['items'])
    else:
        raise ModelFormatError('unsupported schema type')
    if 'enum' in schema and value not in schema['enum']:
        raise ModelFormatError('invalid enum value')


def validate_choice(choice, tools):
    """Return parsed calls; never repair/invent arguments or execute a valid prefix."""
    if choice.get('finish_reason') not in ('stop', 'tool_calls', None):
        raise ModelFormatError('incomplete/filtered completion: ' + str(choice.get('finish_reason'))[:40])
    message = choice.get('message')
    if not isinstance(message, dict) or message.get('role') != 'assistant':
        raise ModelFormatError('missing assistant message')
    calls = message.get('tool_calls') or []
    if not isinstance(calls, list) or len(calls) > 1:
        raise ModelFormatError('submit at most one tool call; inspect its result before the next')
    schemas = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
    parsed = []
    for call in calls:
        if not isinstance(call, dict) or call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id']:
            raise ModelFormatError('invalid tool call envelope')
        function = call.get('function', {})
        if not isinstance(function, dict) or function.get('name') not in schemas:
            raise ModelFormatError('unknown tool')
        raw = function.get('arguments')
        if not isinstance(raw, str):
            raise ModelFormatError('arguments must be a JSON string')
        try:
            args = json.loads(raw, object_pairs_hook=_object)
        except (ValueError, RecursionError):
            raise ModelFormatError('arguments are not valid complete JSON') from None
        _validate(args, schemas[function['name']])
        parsed.append((call, function['name'], args))
    return parsed


class FormatBudget:
    """All attempts consume the existing model-turn budget; allow two retries."""
    def __init__(self, retries=2):
        self.retries = retries
        self.consecutive = 0

    def reject(self):
        self.consecutive += 1
        return self.consecutive <= self.retries

    def accept(self):
        self.consecutive = 0
