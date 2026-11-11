"""Select only this repository's game and its reviewed builder."""
from __future__ import annotations
import importlib
import json
from pathlib import Path, PurePosixPath
import platform_tools

ROOT = Path(__file__).resolve().parent
GAME = 'jak1'
GAMES = (GAME,)
MODULES = {GAME: 'jak1.build_release'}
COMPATIBILITY = {GAME: 'jak1/compatibility.json'}
DEFAULT = {'schemaVersion': 1, 'game': 'jak1', 'displayName': 'Jak and Daxter: The Cinematic Legacy', 'modName': 'jak1-cinematic-models-windows-x64', 'builderModule': 'jak1.build_release', 'compatibilityFile': 'jak1/compatibility.json', 'donorGame': 'jak2', 'runtimeFolders': ['out/jak1/fr3', 'out/jak1/iso', 'game', 'goal_src', 'decompiler', 'custom_assets'], 'thumbnail': 'artwork/jak1.png', 'installGuide': 'LAUNCHER-INSTALL.txt', 'description': 'Cinematic character models in Jak and Daxter gameplay'}


def runtime_folders(game=GAME):
    if game != GAME:
        raise ValueError('Unsupported game')
    return (f'out/{game}/fr3', f'out/{game}/iso', 'game', 'goal_src', 'decompiler', 'custom_assets')


def relative_file(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe package resource path')
    result = Path(root) / path
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Package resource escapes its source directory')
    return result


def validate(value):
    result = dict(value)
    if result.get('schemaVersion') != 1 or result.get('game') != GAME:
        raise ValueError('Unsupported game configuration')
    if result.get('builderModule') != MODULES[GAME] or result.get('compatibilityFile') != COMPATIBILITY[GAME]:
        raise ValueError('Game configuration selects an unsupported builder or compatibility file')
    if tuple(result.get('runtimeFolders', ())) != runtime_folders():
        raise ValueError('Unsupported game runtime folders')
    if result.get('donorGame') not in (None, 'jak2'):
        raise ValueError('Unsupported cinematic donor game')
    if result.get('platform', 'windows-x64') not in platform_tools.PLATFORMS:
        raise ValueError('Unsupported package platform')
    for key in ('displayName', 'modName'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError('Missing game display name or mod name')
    if any(c in result['modName'] for c in '/\\:') or result['modName'].startswith('_'):
        raise ValueError('Unsafe mod name')
    for key in ('thumbnail', 'installGuide'):
        relative_file(ROOT, result[key])
    result.setdefault('description', DEFAULT['description'])
    return result


def default_config(game=None):
    if game not in (None, GAME):
        raise ValueError('Unsupported game')
    return validate(json.loads(json.dumps(DEFAULT)))


def load(root=ROOT, game=None):
    if game not in (None, GAME):
        raise ValueError('Unsupported game')
    path = Path(root) / ('game-config.json' if game is None else f'game-configs/{GAME}.json')
    if not path.is_file():
        raise ValueError('Missing game configuration: ' + str(path))
    return validate(json.loads(path.read_text()))


def builder(config):
    return importlib.import_module(validate(config)['builderModule'])


def compatibility(root, config):
    selected = validate(config)
    value = json.loads(relative_file(root, selected['compatibilityFile']).read_text())
    return platform_tools.select(value, selected['platform']) if 'platform' in selected else value
