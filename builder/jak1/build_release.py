"""Build a Jak 1 launcher payload exclusively from the user's local discs.

The temporary project contains source and the selected extraction inputs, not
a second runnable installation. Jak II supplies the four cinematic donors.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(RELEASE))
import platform_tools
PAYLOAD_PATHS = ['data/out/jak1/fr3/GAME.fr3',
                 'data/out/jak1/fr3/village1.fr3',
                 'data/out/jak1/iso/GAME.CGO']
TARGET_ARCHIVES = ['CGO/KERNEL.CGO', 'CGO/GAME.CGO',
                   'CGO/COMMON.CGO', 'DGO/VI1.DGO']
DONOR_ARCHIVES = ['CGO/KERNEL.CGO', 'CGO/GAME.CGO', 'CGO/COMMON.CGO',
                  'DGO/INTROCST.DGO', 'DGO/HIPHOG.DGO', 'DGO/LINTCSTB.DGO']
SOURCE_CLOSURE = ['jak1/build_release.py', 'jak1/compatibility.json',
    'platform_tools.py', 'platforms/macos-arm64.json',
    'game-configs/jak1.json', 'jak1/jak1_paths.py', 'jak1/rig_validation.py',
    'jak1/build_prototypes.py', 'jak1/keira_feet.py', 'jak1/refit_keira_cuffs.py',
    'jak1/native_io.py', 'jak1/verify_import.py', 'jak1/build_materials.py',
    'jak1/fit_faces.py', 'jak1/build_faces.py', 'jak1/eye_alignment.py',
    'jak1/native_jaw.py', 'jak1/native_face_rig.py', 'jak1/build_textures.py', 'jak1/patch_runtime.py',
    'retarget_glb.py', 'requirements.txt', 'stages/fr3_models.py',
    'stages/inspect_faces.py', 'stages/fix_faces.py',
    'stages/inspect_transformations.py', 'stages/dax_effects.py']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_hash(folder):
    digest = hashlib.sha256()
    files = sorted((p for p in folder.rglob('*') if p.is_file()),
                   key=lambda p: p.relative_to(folder).as_posix())
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode()+b'\0'+
                      sha(path).encode()+b'\n')
    return dict(files=len(files), sha256=digest.hexdigest())


def read_jsonc(path):
    text = re.sub(r'/\*.*?\*/', '', path.read_text(), flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    return json.loads(re.sub(r',\s*([}\]])', r'\1', text))


def disc_folder(root, game):
    root = Path(root).resolve()
    if (root/'SYSTEM.CNF').is_file():
        return root
    return root/'iso_data'/game


def serial_for(disc):
    boot = (disc/'SYSTEM.CNF').read_text()
    match = re.search(r'([A-Z]{4})[_-](\d{3})\.(\d{2})', boot)
    if not match:
        raise ValueError('Cannot identify the disc executable in SYSTEM.CNF')
    return match[1]+'-'+match[2]+match[3], match[1]+'_'+match[2]+'.'+match[3]


def resolved_config(data, config):
    result = dict(config)
    result['sourceRoot'] = str(Path(config.get('sourceData', data)).resolve())
    target = disc_folder(config.get('discData', data), 'jak1')
    donor = disc_folder(config.get('donorData', data), 'jak2')
    for key, path, profiles in [('target', target, config['regionalProfiles']),
                                 ('donor', donor, config['donorProfiles'])]:
        if not (path/'SYSTEM.CNF').is_file():
            raise ValueError(f'Missing local {"Jak 1" if key == "target" else "Jak II cinematic donor"} extraction: {path}')
        serial, executable = serial_for(path)
        if serial not in profiles:
            raise ValueError(f'Unsupported {key} disc revision: {serial}')
        result[key+'Disc'] = str(path)
        result[key+'Serial'] = serial
        result[key+'Executable'] = executable
    result['selectedSerial'] = result['targetSerial']
    return result


def preflight(data: Path, tools: Path, config: dict):
    if sys.flags.optimize or sys.version_info < (3, 11):
        raise ValueError('Use Python 3.11 or newer without -O')
    for line in (RELEASE/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        package, expected = line.split('==')
        try:
            found = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found = 'not installed'
        if found != expected:
            raise ValueError(f'{package}=={expected} is required; found {found}')
    import numpy, scipy, PIL, zstandard  # Validate binary dependencies now.
    config = resolved_config(Path(data), config)
    for name, expected in config['tools'].items():
        path = Path(tools)/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Unsupported OpenGOAL {config["openGoalVersion"]} tool: {name}')
    source = Path(config['sourceRoot'])
    for relative, expected in config['sourceTrees'].items():
        if not (source/relative).is_dir() or tree_hash(source/relative) != expected:
            raise ValueError(f'Unsupported or modified source tree: {relative}')
    for role in ('target', 'donor'):
        profiles = config['regionalProfiles' if role == 'target' else 'donorProfiles']
        profile = profiles[config[role+'Serial']]
        if type(profile.get('elfHash')) is not int:
            raise ValueError(f'Missing pinned {role} executable metadata')
        disc = Path(config[role+'Disc'])
        for relative, expected in profile['inputs'].items():
            if not (disc/relative).is_file() or sha(disc/relative) != expected:
                raise ValueError(f'Unsupported or modified {role} disc input: {relative}')
        metadata = disc/'buildinfo.json'
        if metadata.exists():
            rows = json.loads(metadata.read_text())
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get('serial') != config[role+'Serial']:
                raise ValueError(f'Invalid {role} OpenGOAL disc metadata')
            if profile.get('elfHash') is not None and rows[0].get('elf_hash') != profile['elfHash']:
                raise ValueError(f'Unsupported {role} executable revision')
    if config.get('game') != 'jak1' or config['payloadPaths'] != PAYLOAD_PATHS:
        raise ValueError('Jak 1 payload definition differs from the constructor')
    return config


def run(command, log, env):
    print(log.stem, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{log.name} failed ({result.returncode}):\n'+
                           log.read_text(errors='replace')[-5000:])


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_disc(source, destination, profile, serial):
    """Supply validated game metadata even if an older extraction omitted it.

    The official extractor otherwise silently defaults to Jak 1 black label,
    including when --game jak2 is supplied. Original inputs remain untouched.
    """
    for relative in profile['inputs']:
        link_or_copy(source/relative, destination/relative)
    (destination/'buildinfo.json').write_text(json.dumps([
        dict(serial=serial, elf_hash=profile['elfHash'])])+'\n')
    return destination


def filtered_village(source, destination, source_root):
    """Export only the needed NPC rigs, retaining their original object bytes."""
    art = {row[1] for row in json.loads(
        (source_root/'goal_src/jak1/build/all_objs.json').read_text())
        if row[0].endswith('-ag')}
    raw = source.read_bytes()
    cursor = 64
    records = []
    for _ in range(struct.unpack_from('<I', raw)[0]):
        size = struct.unpack_from('<I', raw, cursor)[0]
        name = raw[cursor+4:cursor+64].split(b'\0')[0].decode()
        end = cursor+64+((size+15)&~15)
        if end > len(raw):
            raise ValueError('Truncated village archive')
        if name not in art or name in ('assistant', 'sage'):
            records.append(raw[cursor:end])
        cursor = end
    if cursor != len(raw):
        raise ValueError('Village archive trailing data differs')
    header = bytearray(raw[:64])
    struct.pack_into('<I', header, 0, len(records))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header+b''.join(records))


def extract(project, tools, disc, game, archives, levels, rip, log, env,
            config, runtime_metadata=False):
    inputs = read_jsonc(project/f'decompiler/config/{game}/ntsc_v1/inputs.jsonc')
    inputs.update(dgo_names=archives, object_file_names=[], str_file_names=[],
                  str_texture_file_names=[], streamed_audio_file_names=[],
                  levels_to_extract=levels)
    input_file = log.with_suffix('.inputs.json')
    input_file.write_text(json.dumps(inputs, indent=2)+'\n')
    override = dict(inputs_file=input_file.as_posix(), rip_levels=rip,
                    dump_objs=runtime_metadata, process_game_text=runtime_metadata,
                    process_game_count=runtime_metadata,
                    process_part_group_table=runtime_metadata,
                    extract_collision=False)
    run([platform_tools.tool_path(tools, 'extractor', config), disc, '--folder', '--game', game,
         '--proj-path', project, '--decompile', '--decomp-config-override',
         json.dumps(override), '--disable-ansi', *platform_tools.compiler_options(config)], log, env)


def compile_command(tools, project, form, config):
    profile = config['regionalProfiles'][config['targetSerial']]
    territory, full_game = profile['territory'], profile['fullGame']
    if type(territory) is not int or territory not in (0, 1) or type(full_game) is not bool:
        raise ValueError('Unsupported Jak 1 compiler edition flags')
    native_gp = project/'goal_src/jak1/game.gp'
    native_body = native_gp.read_bytes()
    iso = (project/'iso_data/jak1').as_posix()
    # MakeSystem and the compiler have separate GOOS environments. Match the
    # official extractor's setters in both, without editing the native GP body.
    flag = '#t' if full_game else '#f'
    prefix = (f'(define *jak1-full-game* {flag})\n'
              f'(define *iso-data* {json.dumps(iso)})\n'
              '(define *use-iso-data-path* #t)\n').encode()
    regional_gp = project.parent/'regional-game.gp'
    regional_gp.write_bytes(prefix+native_body)
    (project.parent/'compiler-edition.json').write_text(json.dumps(dict(
        serial=config['targetSerial'], territory=territory, fullGame=full_game,
        nativeProjectSHA256=sha(native_gp), wrapperSHA256=sha(regional_gp),
        nativeProjectBodyByteExact=regional_gp.read_bytes()[len(prefix):] == native_body,
        makeSystemPrefix=prefix.decode()), indent=2)+'\n')
    command = (f'(begin (seval (set! *default-territory* {territory})) '
               f'(seval (set! *jak1-full-game* {flag})) '
               f'(load-project {json.dumps(regional_gp.as_posix())}) {form})')
    return [platform_tools.tool_path(tools, 'goalc', config), '--game', 'jak1', '--proj-path', project,
            '--disable-ansi', '--port', str(config.get('compilerPort', 8283)),
            '--debug-port', str(config.get('debugPort', 8230)),
            *platform_tools.compiler_options(config), '--cmd', command]


def replace_bones(original, replacement, output):
    raw = original.read_bytes()
    cursor = 64
    rebuilt = bytearray(raw[:64])
    changed = []
    count = struct.unpack_from('<I', raw)[0]
    for _ in range(count):
        size = struct.unpack_from('<I', raw, cursor)[0]
        header = raw[cursor:cursor+64]
        name = header[4:].split(b'\0')[0].decode()
        end = cursor+64+((size+15)&~15)
        if end > len(raw):
            raise ValueError('Truncated native GAME.CGO')
        if name == 'bones' and raw[cursor+64:cursor+68] == b'GOAL':
            body = replacement.read_bytes()
            rebuilt.extend(struct.pack('<I', len(body))+header[4:]+body+
                           b'\0'*(-len(body)%16))
            changed.append(name)
        else:
            rebuilt.extend(raw[cursor:end])
        cursor = end
    if cursor != len(raw) or changed != ['bones']:
        raise ValueError('Native archive code replacement did not match exactly')
    output.write_bytes(rebuilt)
    return dict(nativeObjectCount=count, changedObjects=changed,
                unrelatedObjectRecordsByteExact=True,
                nativeArchiveSHA256=sha(original), outputSHA256=sha(output))


def build(data: Path, tools: Path, work: Path, output: Path, config: dict):
    data, tools, work, output = [Path(p).resolve() for p in (data, tools, work, output)]
    config = preflight(data, tools, config)
    protected = [data, tools, Path(config['sourceRoot']),
                 Path(config['targetDisc']), Path(config['donorDisc'])]
    for path in (work, output):
        if path.exists():
            raise ValueError(f'Use a new destination: {path}')
        if any(path == p or path.is_relative_to(p) or p.is_relative_to(path)
               for p in protected):
            raise ValueError('Build/output directories must be outside every input')
    if work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError('Work and payload folders must be separate')
    parent = work.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < config.get('minimumFreeBytes', 2*1024**3):
        raise ValueError('At least 2 GiB of free build space is required')
    work.mkdir(parents=True)
    project = work/'extraction'
    project.mkdir()
    source = Path(config['sourceRoot'])
    for folder in ('goal_src', 'decompiler', 'game'):
        shutil.copytree(source/folder, project/folder)
    for folder in ('levels', 'models'):
        custom = source/'custom_assets/jak1'/folder
        if custom.exists():
            shutil.copytree(custom, project/'custom_assets/jak1'/folder)
    # The input directory is selected, never a cloned runnable installation.
    iso = project/'iso_data/jak1'
    disc = stage_disc(Path(config['targetDisc']), iso,
        config['regionalProfiles'][config['targetSerial']], config['targetSerial'])
    donor = stage_disc(Path(config['donorDisc']), work/'donor-input',
        config['donorProfiles'][config['donorSerial']], config['donorSerial'])
    env = dict(os.environ, CINEMATIC_BUILD_WORK=str(work),
               CINEMATIC_JAK1_WORK=str(work), CINEMATIC_JAK1_DATA=str(project))
    # Load the original global texture database before selecting replacements.
    # Selected native archives are hardlinked where possible, never modified.
    extract(project, tools, disc, 'jak1', config.get('nativeArchives', TARGET_ARCHIVES), ['VI1.DGO'], False,
            work/'01-native.log', env, config, runtime_metadata=True)
    for name in ('GAME', 'village1'):
        shutil.copy2(project/f'out/jak1/fr3/{name}.fr3', work/f'{name}-original.fr3')
    run(compile_command(tools, project, '(make "$OUT/iso/GAME.CGO")', config),
        work/'02-native-code.log', env)
    shutil.copy2(project/'out/jak1/iso/GAME.CGO', work/'GAME-original.CGO')
    selective = work/'native-rigs'
    for relative in TARGET_ARCHIVES[:-1]+['SYSTEM.CNF', config['targetExecutable']]:
        link_or_copy(disc/relative, selective/relative)
    if (disc/'buildinfo.json').exists():
        link_or_copy(disc/'buildinfo.json', selective/'buildinfo.json')
    filtered_village(disc/'DGO/VI1.DGO', selective/'DGO/VI1.DGO', source)
    extract(project, tools, selective, 'jak1', TARGET_ARCHIVES, ['VI1.DGO'], True,
            work/'03-native-rigs.log', env, config)
    extract(project, tools, donor, 'jak2', DONOR_ARCHIVES,
            ['INTROCST.DGO', 'HIPHOG.DGO', 'LINTCSTB.DGO'], True,
            work/'04-donors.log', env, config)
    def stage(name, *args):
        run([sys.executable, HERE/name, *args], work/(name[:-3]+'.log'), env)
    stage('build_prototypes.py')
    destination = project/'custom_assets/jak1/merc_replacements'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('eichar-lod0', 'sidekick-lod0', 'sage-lod0', 'assistant-lod0'):
        shutil.copy2(work/f'rig-prototypes/{name}.glb', destination/f'{name}.glb')
    extract(project, tools, disc, 'jak1', TARGET_ARCHIVES, ['VI1.DGO'], False,
            work/'05-import.log', env, config)
    for name in ('GAME', 'village1'):
        shutil.copy2(project/f'out/jak1/fr3/{name}.fr3', work/f'{name}-import.fr3')
    for name in ('build_materials.py', 'fit_faces.py', 'build_faces.py', 'build_textures.py'):
        stage(name)
    stage('patch_runtime.py', '--data', source, '--project', project,
          '--materials', work/'material-report.json')
    # Use upstream GAME ordering: the isolated bones dependency graph reaches
    # pad.gc before its PC settings declarations. Only bones is packed below.
    run(compile_command(tools, project, '(make "$OUT/iso/GAME.CGO")', config),
        work/'06-cinematic-code.log', env)
    archive_report = replace_bones(work/'GAME-original.CGO',
        project/'out/jak1/obj/bones.o', work/'GAME-cinematic.CGO')
    output.mkdir(parents=True)
    for relative, source_file in zip(PAYLOAD_PATHS,
            ['GAME-cinematic.fr3', 'village1-cinematic.fr3', 'GAME-cinematic.CGO']):
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work/source_file, destination)
    manifest = dict(schemaVersion=1, game='jak1', version=config['version'],
        openGoalVersion=config['openGoalVersion'], platform=config['platform'],
        displayName='Jak and Daxter: The Cinematic Legacy', author='@jacopobassan',
        website='https://jacopobassan.me', runtimeSha256=config['tools'][platform_tools.tool_path(tools, 'gk', config).name],
        privateGeneratedPayload=True, releaseReady=False,
        sourceDiscSerial=config['targetSerial'], donorDiscSerial=config['donorSerial'],
        features=dict(models=['Jak', 'Daxter', 'Samos', 'Keira'],
                      nativeLiveEyes=True, nativeFacialChannels=True,
                      nativeOpenFeet=True, nativeCuffArticulation=True),
        files=[dict(path=p, sha256=sha(output/p), bytes=(output/p).stat().st_size)
               for p in PAYLOAD_PATHS])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    validation = dict(status='Offline construction complete; runtime scope is recorded separately',
        sourceDiscSerial=config['targetSerial'], donorDiscSerial=config['donorSerial'],
        archive=archive_report, payload=manifest['files'],
        compilerEdition=json.loads((work/'compiler-edition.json').read_text()),
        sources={p: sha(RELEASE/p) for p in SOURCE_CLOSURE},
        noInstalledFilesChanged=True, noGameAssetsInSource=True)
    (work/'build-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', '--game-data', dest='data', type=Path, required=True)
    for name in ('tools', 'work', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE/'compatibility.json')
    parser.add_argument('--platform', choices=platform_tools.PLATFORMS,
                        help='Native tool platform; defaults to the current machine')
    parser.add_argument('--source-data', type=Path)
    parser.add_argument('--disc-data', type=Path)
    parser.add_argument('--donor-data', type=Path)
    args = parser.parse_args()
    config = platform_tools.select(json.loads(args.config.read_text()), args.platform)
    for option in ('source_data', 'disc_data', 'donor_data'):
        value = getattr(args, option)
        if value is not None:
            head, tail = option.split('_')
            config[head+tail.title()] = str(value.resolve())
    result = build(args.data, args.tools, args.work, args.output, config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
