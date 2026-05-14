use serde_json::Value;

/// Remove JSON Schema keywords unsupported by Gemini/Vertex AI function calling.
const BANNED: &[&str] = &[
    "$schema",
    "$ref",
    "propertyNames",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "const",
    "additionalProperties",
    "default",
    "examples",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "minProperties",
    "maxProperties",
];

pub fn clean(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let cleaned: serde_json::Map<String, Value> = map
                .iter()
                .filter(|(k, _)| !BANNED.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), clean(v)))
                .collect();
            Value::Object(cleaned)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(clean).collect()),
        other => other.clone(),
    }
}
