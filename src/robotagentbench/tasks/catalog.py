"""Load immutable registration metadata without importing task implementations."""
from dataclasses import dataclass
from importlib.resources import files
import json
import re

CATALOG_NAMES = ("standalone", "openend")
ENTRY_KINDS = {"standalone": "task types", "openend": "component types"}
SCHEMA = "robotagentbench.catalog.v1"
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,127}\Z", re.ASCII)


class CatalogError(ValueError):
    """A catalog is malformed or does not identify a supported namespace."""


@dataclass(frozen=True)
class Catalog:
    name: str
    entry_kind: str
    identifiers: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA,
            "catalog": self.name,
            "entry_kind": self.entry_kind,
            "count": len(self.identifiers),
            "identifiers": list(self.identifiers),
        }


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError("Duplicate catalog field")
        result[key] = value
    return result


def parse_catalogs(text: str) -> dict[str, Catalog]:
    try:
        value = json.loads(text, object_pairs_hook=_unique_keys)
    except (ValueError, TypeError) as exc:
        raise CatalogError("Invalid catalog JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "catalogs"}:
        raise CatalogError("Unexpected catalog fields")
    if value["schema_version"] != SCHEMA:
        raise CatalogError("Unsupported catalog schema")
    catalogs = value["catalogs"]
    if not isinstance(catalogs, dict) or set(catalogs) != set(CATALOG_NAMES):
        raise CatalogError("Unexpected catalog namespaces")
    result = {}
    for name, identifiers in catalogs.items():
        if not isinstance(identifiers, list) or not identifiers:
            raise CatalogError("A catalog needs a nonempty identifier list")
        if any(not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier)
               for identifier in identifiers):
            raise CatalogError("Invalid task identifier")
        if len(set(identifiers)) != len(identifiers):
            raise CatalogError("Duplicate task identifier")
        result[name] = Catalog(name, ENTRY_KINDS[name], tuple(identifiers))
    return result


def load_catalog(name: str) -> Catalog:
    if name not in CATALOG_NAMES:
        raise CatalogError("Unknown catalog namespace")
    text = files("robotagentbench.tasks").joinpath("catalogs.json").read_text(encoding="utf-8")
    return parse_catalogs(text)[name]
