"""Persistent local demo profiles. Demo member IDs are not authentication."""
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class ProfileStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS outfit_views (user_id TEXT NOT NULL, outfit_id TEXT NOT NULL, viewed_at REAL NOT NULL, PRIMARY KEY(user_id, outfit_id))')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def validate_user(uid):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', uid):
            raise ValueError('회원 ID가 올바르지 않습니다.')

    def get(self, uid):
        self.validate_user(uid)
        with self.connect() as db:
            row = db.execute('SELECT data FROM profiles WHERE user_id=?', (uid,)).fetchone()
        return json.loads(row[0]) if row else {'owned': [], 'styles': []}

    def update(self, uid, action, value):
        self.validate_user(uid)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM profiles WHERE user_id=?', (uid,)).fetchone()
            data = json.loads(row[0]) if row else {'owned': [], 'styles': []}
            if action == 'add':
                data['owned'] = list(dict.fromkeys(data['owned'] + value))
                if len(data['owned']) > 500:
                    raise ValueError('로컬 옷장은 최대 500개까지 보관합니다.')
            elif action == 'remove':
                data['owned'] = [pid for pid in data['owned'] if pid not in value]
            elif action == 'styles':
                data['styles'] = list(dict.fromkeys(value))
            elif action == 'audience':
                if len(value) != 1 or value[0] not in ('all', 'women', 'men', 'unisex', 'kids'):
                    raise ValueError('쇼핑 대상을 확인해 주세요.')
                data['shopping_audience'] = value[0]
            else:
                raise ValueError('지원하지 않는 변경입니다.')
            db.execute('INSERT INTO profiles VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET data=excluded.data',
                       (uid, json.dumps(data, ensure_ascii=False)))
        return data

    def record_outfit_view(self, uid, oid):
        self.validate_user(uid)
        with self.connect() as db:
            db.execute('INSERT INTO outfit_views VALUES (?, ?, ?) ON CONFLICT(user_id, outfit_id) DO UPDATE SET viewed_at=excluded.viewed_at',
                       (uid, oid, time.time()))

    def save_survey(self, uid, selected, skipped=False, dataset_version=None):
        self.validate_user(uid)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM profiles WHERE user_id=?', (uid,)).fetchone()
            data = json.loads(row[0]) if row else {'owned': [], 'styles': []}
            prior = data.get('survey', {})
            now = time.time()
            data['survey'] = {'selected': prior.get('selected', []) if skipped else selected,
                              'completed': True, 'skipped': skipped, 'updated_at': now,
                              'revision': prior.get('revision', 0) + 1,
                              'selected_at': prior.get('selected_at', prior.get('updated_at')) if skipped else now,
                              'dataset_version': prior.get('dataset_version') if skipped else dataset_version}
            db.execute('INSERT INTO profiles VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET data=excluded.data',
                       (uid, json.dumps(data, ensure_ascii=False)))
        return data['survey']

    def set_registration(self, uid, info):
        self.validate_user(uid)
        if not isinstance(info, dict) or set(info) != {'gender', 'age_group'}:
            raise ValueError('가입 정보의 성별과 연령대를 확인해 주세요.')
        if info['gender'] not in ('F', 'M', 'unknown') or info['age_group'] not in ('10s', '20s', '30s', '40s', '50s', '60s+', 'unknown'):
            raise ValueError('성별 또는 연령대 값이 올바르지 않습니다.')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM profiles WHERE user_id=?', (uid,)).fetchone()
            data = json.loads(row[0]) if row else {'owned': [], 'styles': []}
            data['registration'] = dict(info)
            db.execute('INSERT INTO profiles VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET data=excluded.data',
                       (uid, json.dumps(data, ensure_ascii=False)))
        return data

    def cohort_views(self, uid, now=None):
        """Observed peers only; never manufacture demographic histories."""
        self.validate_user(uid)
        account = self.get(uid).get('registration', {})
        if account.get('gender', 'unknown') == 'unknown' or account.get('age_group', 'unknown') == 'unknown':
            return {}
        now = time.time() if now is None else now
        with self.connect() as db:
            rows = db.execute('SELECT p.user_id, p.data, v.outfit_id, v.viewed_at FROM profiles p JOIN outfit_views v ON p.user_id=v.user_id WHERE p.user_id<>? AND v.viewed_at>=? ORDER BY v.viewed_at DESC',
                              (uid, now - 90 * 86400)).fetchall()
        result = {}
        for peer, raw, oid, at in rows:
            if json.loads(raw).get('registration') != account:
                continue
            views = result.setdefault(peer, [])
            if len(views) < 50:
                views.append({'outfit_id': oid, 'viewed_at': at})
        return result

    def recent_outfit_views(self, uid, now=None):
        self.validate_user(uid)
        now = time.time() if now is None else now
        with self.connect() as db:
            rows = db.execute('SELECT outfit_id, viewed_at FROM outfit_views WHERE user_id=? AND viewed_at>=? ORDER BY viewed_at DESC, outfit_id LIMIT 50',
                              (uid, now - 90 * 86400)).fetchall()
        return [{'outfit_id': oid, 'viewed_at': at} for oid, at in rows]
