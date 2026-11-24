"""Мини-раннер задач на арендованной машине (вместо SSH). Все запросы требуют заголовок X-Token == $RUNNER_TOKEN.
POST /upload?path=/root/work/x.py   (тело = содержимое файла)
POST /run?name=job1                 (тело = bash-команда; запускается в фоне, лог /root/logs/job1.log)
GET  /status?name=job1              (running/exit code + хвост лога)
GET  /file?path=/root/work/out.zip  (скачать файл)
"""
import http.server, os, subprocess, urllib.parse, json

TOKEN = os.environ.get("RUNNER_TOKEN", "")
JOBS = {}


class H(http.server.BaseHTTPRequestHandler):
    def _ok(self, code=200, body=b"", ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def _auth(self):
        if not TOKEN or self.headers.get("X-Token") != TOKEN:
            self._ok(403, b'{"error":"forbidden"}'); return False
        return True

    def do_POST(self):
        if not self._auth(): return
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/upload":
            os.makedirs(os.path.dirname(q["path"]), exist_ok=True)
            open(q["path"], "wb").write(data); self._ok(body=b'{"ok":true}')
        elif u.path == "/run":
            name = q.get("name", "job"); log = open(f"/root/logs/{name}.log", "ab")
            JOBS[name] = subprocess.Popen(["bash", "-lc", data.decode()], stdout=log, stderr=subprocess.STDOUT, cwd="/root")
            self._ok(body=json.dumps({"started": name}).encode())
        else:
            self._ok(404)

    def do_GET(self):
        if not self._auth(): return
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/status":
            name = q.get("name", "job"); p = JOBS.get(name)
            tail = b""
            try:
                with open(f"/root/logs/{name}.log", "rb") as f:
                    f.seek(0, 2); f.seek(max(0, f.tell() - 3000)); tail = f.read()
            except FileNotFoundError:
                pass
            st = "unknown" if p is None else ("running" if p.poll() is None else f"exit {p.returncode}")
            self._ok(body=json.dumps({"status": st, "tail": tail.decode(errors="replace")}).encode())
        elif u.path == "/file":
            try:
                self._ok(body=open(q["path"], "rb").read(), ctype="application/octet-stream")
            except FileNotFoundError:
                self._ok(404)
        else:
            self._ok(404)

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", 8190), H).serve_forever()
