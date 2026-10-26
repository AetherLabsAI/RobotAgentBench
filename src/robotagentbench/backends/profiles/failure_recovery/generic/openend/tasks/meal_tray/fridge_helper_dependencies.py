"""Load the existing helper's installed Isaac API dependency, no settings writes."""
def ensure(manifest):
    if not any(o.get('id')=='refrigerator' for o in manifest.get('objects',[])):return
    import importlib
    try:
        importlib.import_module('isaacsim.core.prims')
    except ModuleNotFoundError as exc:
        if exc.name not in ('isaacsim','isaacsim.core','isaacsim.core.prims'):raise
        import omni.kit.app
        manager=omni.kit.app.get_app().get_extension_manager()
        # This installed image keeps the original dynamic_control dependency
        # in its deprecated-extension folder, outside the minimal CPU app paths.
        from pathlib import Path
        installed=Path('/isaac-sim/extsDeprecated')
        if installed.is_dir():manager.add_path(str(installed))
        if not manager.set_extension_enabled_immediate('isaacsim.core.prims',True):
            raise RuntimeError('Existing refrigerator helper requires installed isaacsim.core.prims')
        importlib.import_module('isaacsim.core.prims')
