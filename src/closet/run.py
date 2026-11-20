"""Portable foreground launcher. Ctrl+C stops only this launcher's children."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parents[2]
from src.closet.runtime_paths import data_root, data_demo, config_path, media_path, required_files


def preflight(test=False, images=False):
    missing = [p for p in required_files(test) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing external data:\n' + '\n'.join(str(p) for p in missing))
    import numpy
    print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, flush=True)
    if images:
        cfg = json.loads(config_path().read_text(encoding='utf-8'))
        if cfg.get('demo_mode') == 'curated_snapshot':
            snapshot = json.loads((data_demo()/cfg['snapshot_file']).read_text(encoding='utf-8'))
            paths = {media_path(row['path']) for row in snapshot['images'].values()}
        else:
            locations = [(data_demo()/cfg['outfits_dir']/'queries.json','image_path'),
                (data_demo()/cfg['outfits_dir']/'outfits.json','original_path'),
                (data_demo()/cfg['products_dir']/'product_manifest.json','path')]
            paths = {media_path(row[field]) for f,field in locations for row in json.loads(f.read_text(encoding='utf-8'))}
        absent = [p for p in paths if not p.is_file()]
        if absent:
            raise FileNotFoundError(f'{len(absent)} images missing; first paths:\n' + '\n'.join(map(str, absent[:10])))
        print('Image files present:', len(paths), flush=True)
    print('External data files present. Frozen hashes are checked by retrieval startup.', flush=True)


def available(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as e:
            raise RuntimeError(f'Port {port} is in use. Select other --retrieval-port/--shop-port values.') from e


def wait_ready(process, port):
    for _ in range(150):
        if process.poll() is not None:
            raise RuntimeError(f'Server exited on port {port}. See the error above.')
        try:
            with urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.2)
    raise RuntimeError(f'Server startup timed out on port {port}')


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=Path, default=HERE / 'configs/closet_demo.json')
    config_args, _ = config_parser.parse_known_args()
    defaults = json.loads(config_args.config.read_text(encoding='utf-8'))
    ap = argparse.ArgumentParser(parents=[config_parser])
    ap.add_argument('--data-root', type=Path, default=HERE / defaults['data_root'])
    ap.add_argument('--retrieval-port', type=int, default=defaults['retrieval_port'])
    ap.add_argument('--shop-port', type=int, default=defaults['shop_port'])
    ap.add_argument('--team-url', default=defaults['team_url'])
    ap.add_argument('--state-dir', type=Path, default=HERE / defaults['state_dir'])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--check-images', action='store_true')
    ap.add_argument('--test', action='store_true', help='Run the integration suite for the supplied data mode with a temporary DB')
    args = ap.parse_args()
    args.state_dir = args.state_dir.expanduser().resolve()
    if not all(1 <= n <= 65535 for n in [args.retrieval_port,args.shop_port]):
        ap.error('Port must be in 1..65535')
    if args.retrieval_port == args.shop_port:
        ap.error('The two server ports must be different')
    os.environ['CLOSET_DATA_ROOT'] = str(args.data_root.expanduser().resolve())
    os.environ['CLOSET_RETRIEVAL_URL'] = f'http://127.0.0.1:{args.retrieval_port}'
    os.environ['CLOSET_TEAM_URL'] = args.team_url.rstrip('/')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    preflight(args.test, args.check_images)
    if args.check:
        return
    available(args.retrieval_port)
    if not args.test:
        available(args.shop_port)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    children = []
    def launch(module, *extra, capture=False):
        p = subprocess.Popen([sys.executable, '-u', '-m', module, *map(str,extra)],
            cwd=HERE, env=os.environ.copy(), creationflags=flags,
            stdout=subprocess.PIPE if capture else sys.stdout,
            stderr=subprocess.STDOUT if capture else sys.stderr,
            text=True, encoding='utf-8', errors='replace')
        children.append(p)
        return p
    try:
        retrieval = launch('src.closet.retrieval','--port',args.retrieval_port)
        wait_ready(retrieval,args.retrieval_port)
        if args.test:
            config = json.loads(config_path().read_text(encoding='utf-8'))
            compact = config.get('demo_mode') == 'curated_snapshot'
            test_file = {'reviewed_dataset':'test_reviewed_dataset.py',
                         'expanded60_dataset':'test_expanded60_dataset.py',
                         'gender_five_dataset':'test_gender_five_dataset.py',
                         'publisher_dataset':'test_publisher_dataset.py',
                         'expanded_dataset':'test_expanded_dataset.py'}.get(config.get('test_suite'), 'test_compact.py') if compact else 'test_integration.py'
            tests = launch('tests.closet.integration.' + Path(test_file).stem, capture=True)
            output, _ = tests.communicate()
            print(output, flush=True)
            raise SystemExit(tests.returncode)
        args.state_dir.mkdir(parents=True,exist_ok=True)
        shop = launch('src.closet.shop.server','--port',args.shop_port,
            '--db',args.state_dir/'profiles.sqlite3')
        wait_ready(shop,args.shop_port)
        print(f'Open http://127.0.0.1:{args.shop_port}/shop/#/mypage', flush=True)
        print('Ctrl+C stops this demo. Data files are read-only; new profiles are saved in --state-dir.', flush=True)
        while True:
            if retrieval.poll() is not None or shop.poll() is not None:
                raise RuntimeError('A demo server stopped unexpectedly')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('\nStopping demo.', flush=True)
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError,RuntimeError,ValueError,ImportError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
