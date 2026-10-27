"""Original raw-action tool schema; units/limits stay native-owner controlled."""
def function(name,description,properties):
    return dict(type='function',function=dict(name=name,description=description,
        parameters=dict(type='object',properties=properties,required=list(properties),additionalProperties=False)))

TOOLS=[function('execute_actions','Execute explicit physical-unit 22D rows at 20 Hz, then observe. No interpolation or IK.',
    {'actions':dict(type='array',minItems=1,maxItems=600,
        items=dict(type='array',minItems=22,maxItems=22,items=dict(type='number')))})]
