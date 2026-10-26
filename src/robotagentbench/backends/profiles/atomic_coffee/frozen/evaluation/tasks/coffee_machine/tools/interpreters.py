"""Interpreters of this host. A frozen snapshot has no virtualenv next to it, so they are named explicitly."""
import os

HOME = '/home/lingjun/robot_agent_arena'
API = os.environ.get('ARENA_API_PYTHON', HOME + '/.venv-api/bin/python')      # public service, runner, launchers
ISAAC = os.environ.get('ARENA_ISAAC_PYTHON', HOME + '/.venv/bin/python')      # Isaac Sim 5.1 worker, probes, CPU USD
