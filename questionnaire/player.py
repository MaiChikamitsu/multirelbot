"""Local video player using the questionnaire's registered stop times."""
import json
import sqlite3
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file

from .storage import validate_config


def create_player(data_dir=None):
    root = Path(data_dir or Path(__file__).parent / 'private').resolve()
    app = Flask(__name__)

    def configs():
        # Read the registered configuration without modifying questionnaire data.
        with sqlite3.connect((root / 'responses.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
            rows = db.execute('SELECT config FROM experiments ORDER BY id').fetchall()
        registered = {cfg['id']: cfg for (raw,) in rows
                      if (cfg := json.loads(raw)).get('modality') != 'text' and not cfg.get('preview')}
        for path in sorted((root / 'player_configs').glob('*.json')):
            cfg = json.loads(path.read_text())
            validate_config(cfg, root / 'videos')
            registered[cfg['id']] = cfg
        return registered

    @app.get('/')
    def index():
        return render_template('player.html')

    @app.get('/api/videos')
    def videos():
        return jsonify([dict(id=c['id'], theme=c['theme'], start=c.get('start_seconds', 0),
                             stops=c['end_seconds']) for c in configs().values()])

    @app.get('/media/<experiment>')
    def media(experiment):
        cfg = configs().get(experiment)
        if cfg is None:
            abort(404)
        return send_file(validate_config(cfg, root / 'videos'), conditional=True)

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; frame-ancestors 'none'"
        return response

    return app


if __name__ == '__main__':
    create_player().run(host='127.0.0.1', port=8893, debug=False)
