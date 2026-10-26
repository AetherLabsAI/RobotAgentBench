import ast
from pathlib import Path
import numpy as np

source = Path(__file__).resolve().parents[1] / "real2sim/production/task8_clear_physx.py"
function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "base_initial_readback")
namespace = {"np": np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
readback = namespace["base_initial_readback"]


def test_measured_settle_error_uses_angular_units_and_records_tolerances():
    r = readback([-3.16611886, .267892897, .306466788], [-3.166118825, .267892892, .307563831], [-3.1748, .271021, 0.], sampled=True)
    assert r["verified"] and .001 < r["yaw_error_deg"] < .002
    assert r["xy_error_m"] < 1e-6


def test_missing_injection_or_excess_motion_rejected():
    nominal = [0., 0., 0.]
    assert not readback(nominal, [.01, .01, .5], nominal, sampled=True)["verified"]
    assert not readback([.01, .01, .5], [.01, .01, .5], nominal, sampled=False)["verified"]
    assert not readback([.012, .01, .5], [.01, .01, .5], nominal, sampled=True)["verified"]
    assert not readback([.01, .01, .52], [.01, .01, .5], nominal, sampled=True)["verified"]
    assert not readback([.01, .01, np.nan], [.01, .01, .5], nominal, sampled=True)["verified"]


def test_yaw_wrap_is_not_physical_error():
    assert readback([.01, .01, -179.999], [.01, .01, 180.001], [0., 0., 179.], sampled=True)["verified"]
