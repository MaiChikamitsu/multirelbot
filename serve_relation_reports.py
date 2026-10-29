"""Serve generated reports locally with persistent exclude/restore controls."""
import argparse
import json
import secrets
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from report_management import exclusion_path, load_exclusions


class ReportManager:
    def __init__(self, config_path, analyze=None):
        self.config_path=Path(config_path).resolve()
        if analyze is None:
            from analyze_relations import analyze
        self.analyze=analyze
        self.token=secrets.token_urlsafe(32)
        cfg=json.loads(self.config_path.read_text())
        self.output=(self.config_path.parent/cfg.get('output','analysis_output')).resolve()

    def update(self, experiment, run, action):
        if action not in ['exclude','restore'] or not isinstance(experiment,str) or not isinstance(run,str):
            raise ValueError('Invalid operation')
        records=load_exclusions(self.config_path)
        key=(experiment,run)
        excluded={(r['experiment'],r['run']) for r in records}
        manifest_path=self.output/'report_manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        known={(r['experiment'],r['run']) for r in manifest}
        if key not in known|excluded:
            raise ValueError('Unknown experiment/run')
        if action=='exclude':excluded.add(key)
        else:excluded.discard(key)
        path=exclusion_path(self.config_path)
        original=path.read_bytes() if path.exists() else None
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps([dict(experiment=e,run=r) for e,r in sorted(excluded)],ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)
        try:
            self.analyze(self.config_path)
        except Exception:
            if original is None:path.unlink()
            else:path.write_bytes(original)
            # Restore generated reports as well as the saved exclusion list.
            try:self.analyze(self.config_path)
            except Exception:pass
            raise


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,manager,**kwargs):
        self.manager=manager
        super().__init__(*args,directory=str(manager.output),**kwargs)

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def json_response(self,status,value):
        payload=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers();self.wfile.write(payload)

    def local_host(self):
        port=self.server.server_port
        return self.headers.get('Host') in [f'127.0.0.1:{port}', f'localhost:{port}']

    def do_GET(self):
        if not self.local_host():self.send_error(403);return
        if urlparse(self.path).path=='/api/status':
            self.json_response(200,dict(token=self.manager.token));return
        # Serve only the generated output tree, not arbitrary source files.
        resolved=Path(self.translate_path(self.path)).resolve()
        if resolved!=self.manager.output and self.manager.output not in resolved.parents:
            self.send_error(403);return
        super().do_GET()

    def do_POST(self):
        if not self.local_host():self.json_response(403,dict(error='Local access only'));return
        if self.path!='/api/run-visibility':self.json_response(404,dict(error='Unknown endpoint'));return
        if not secrets.compare_digest(self.headers.get('X-Report-Token',''),self.manager.token):
            self.json_response(403,dict(error='このレポート画面から操作してください'));return
        if self.headers.get_content_type()!='application/json':
            self.json_response(400,dict(error='JSON request required'));return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length))
            self.manager.update(body.get('experiment'),body.get('run'),body.get('action'))
        except (ValueError,KeyError,TypeError) as error:
            self.json_response(400,dict(error=str(error)));return
        except Exception:
            self.json_response(500,dict(error='再集計に失敗したため除外設定を元に戻しました。サーバーと設定を確認してください。'));return
        self.json_response(200,dict(ok=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--port',type=int,default=8890)
    args=parser.parse_args()
    manager=ReportManager(args.config)
    manager.analyze(manager.config_path)
    server=HTTPServer(('127.0.0.1',args.port),partial(Handler,manager=manager))
    print(f'レポートを開く: http://127.0.0.1:{args.port}/report.html',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
