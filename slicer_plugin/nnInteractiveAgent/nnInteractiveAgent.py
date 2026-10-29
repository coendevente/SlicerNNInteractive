"""nnInteractive Agent: say what to segment, an AI agent segments it with nnInteractive.

This module is a front end of the nnInteractive agent web demo server (``nninteractive-agent-web``
from https://github.com/MIC-DKFZ/nninteractive-agent), used instead of its browser page. The server
runs the agent (Claude Code or Codex) and nnInteractive; Slicer shows the image, the chat and the
segmentation:

- the volume selected in Slicer is uploaded to the server (an image the server already works on is
  downloaded into Slicer);
- messages go to the agent over the server's websocket, and the agent's replies, tool calls and the
  images it looked at come back as chat events;
- the server sends its label layers (committed objects, the object being worked on, its prompts) as
  binary patches in the voxel grid of the image, which are mirrored into segmentation nodes.

Workspace arrays of the server are indexed [axis0, axis1, axis2] = reversed ITK index order, which is
the same as the [k, j, i] order of ``slicer.util.arrayFromVolume``. So label patches are written
directly into arrays with the shape of the volume.
"""

import html
import importlib
import importlib.util
import json
import logging
import os
import queue
import re
import secrets
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zlib

import numpy as np

import ctk
import qt
import slicer
import vtk
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

COOKIE = "nni_session"  # nninteractive_agent.web.auth.COOKIE
DEFAULT_SERVER = "http://127.0.0.1:8443/"
SETTINGS_PREFIX = "nnInteractiveAgent/"
POS_COLOR, NEG_COLOR = "#2bff4a", "#ff3b3b"  # nninteractive_agent.core.render
PROMPT_POS, PROMPT_NEG = 1, 2  # values of the "prompts" layer
OBJECT_SEGMENT_PREFIX = "agent_object_"
CURRENT_SEGMENT, POS_SEGMENT, NEG_SEGMENT = "agent_current", "agent_prompts_positive", "agent_prompts_negative"
IMAGE_ATTRIBUTE = "nnInteractiveAgent.Image"  # "<volume key>:<n>" of the server image a node belongs to
ROLE_ATTRIBUTE = "nnInteractiveAgent.Role"  # "result" or "work"
NOTES_TAG = "nnInteractiveAgent.Notes"  # segment tag: the agent's notes on the object
MAX_CHAT_IMAGES = 4  # thumbnails shown per tool result
# The internal server: nninteractive-agent installed in Slicer's Python (an archive, so no git is needed).
AGENT_REPOSITORY = "MIC-DKFZ/nninteractive-agent"
AGENT_COMMIT = "8fe94bda678d0c09c7c8363261e82bd984802984"  # the recommended version, tested with this module


def agent_archive_url(ref: str) -> str:
    return f"https://github.com/{AGENT_REPOSITORY}/archive/{ref}.zip"


AGENT_PACKAGE = f"nninteractive-agent[web] @ {agent_archive_url(AGENT_COMMIT)}"
AGENT_MODULES = ("nninteractive_agent", "claude_agent_sdk", "fastapi", "uvicorn", "multipart")
LOCAL_INFERENCE_PACKAGE = "nnInteractive>=2.6.0"  # the [local] extra of nninteractive-agent
DEFAULT_INTERNAL_PORT = 8444  # not 8443, the default of nninteractive-agent-web started by hand
LLM_BACKENDS = (  # nninteractive_agent.web.llm.BACKENDS
    ("claude", "Claude subscription (Claude Code login)"),
    ("claude-api", "Claude, Anthropic API key"),
    ("codex", "ChatGPT subscription (Codex login)"),
    ("codex-api", "Codex, OpenAI API key"),
    ("server", "Own model on an Anthropic-compatible server (e.g. vLLM)"),
)
SUBSCRIPTION_BACKENDS = ("claude", "codex")


###############################################################################
# nnInteractiveAgent
###############################################################################


class nnInteractiveAgent(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("nnInteractive Agent")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Andras Lasso"]
        self.parent.helpText = _("""
Describe in plain text what to segment; an AI agent (Claude Code or Codex) segments it with nnInteractive,
checks the result in 2D and 3D and corrects it.<br><br>
The agent runs in an nnInteractive agent server (<code>nninteractive-agent-web</code>). In the Configuration
tab, choose the internal server (Slicer installs it and starts it on this computer) or a remote one (paste the
link it prints at start-up, or its address and password). Then send the selected volume to the agent and type
a request: this connects, and starts the internal server if needed. Committed objects appear in the segmentation
"&lt;volume&gt; agent"; the object the agent is working on and its prompts in "&lt;volume&gt; agent work".
Other segmentations can be selected as outputs in the Image section.<br><br>
The agent's segmentation is a mirror of the server's: edits made in Slicer are not sent back and are replaced
when the agent changes that object again. Clone the segmentation node to edit it.<br><br>
See <a href="https://github.com/MIC-DKFZ/nninteractive-agent">nnInteractive Agent</a> for setting up the server.
""")
        self.parent.acknowledgementText = _("""
nnInteractive Agent: https://github.com/MIC-DKFZ/nninteractive-agent. nnInteractive: Isensee, F., Rokuss, M.,
Kramer, L., et al. (2025), https://arxiv.org/abs/2503.08373.""")


###############################################################################
# Helpers (no Qt or MRML)
###############################################################################


def decode_patch(data: bytes):
    """(header, crop) of a binary label patch: uint32 header length, JSON header, zlib-compressed crop
    (``nninteractive_agent.web.layers.encode_patch``)."""
    (n,) = struct.unpack_from("<I", data)
    header = json.loads(bytes(data[4:4 + n]).decode())
    shape = [hi - lo for lo, hi in header["box"]]
    body = data[4 + n:]
    crop = np.frombuffer(zlib.decompress(body), dtype=np.dtype(header["dtype"])).reshape(shape) if body else None
    return header, crop


def encode_patch(header: dict, crop: np.ndarray) -> bytes:
    """The server's encoding (for tests)."""
    head = json.dumps(header).encode()
    return struct.pack("<I", len(head)) + head + zlib.compress(np.ascontiguousarray(crop).tobytes(), 1)


def parse_server_link(text: str):
    """(base URL ending in '/', key or None) of a link such as http://127.0.0.1:8443/?key=..."""
    text = text.strip() or DEFAULT_SERVER
    if "://" not in text:
        text = "http://" + text
    parts = urllib.parse.urlsplit(text)
    key = urllib.parse.parse_qs(parts.query).get("key", [None])[0]
    path = parts.path
    if path.endswith("/login"):
        path = path[: -len("login")]
    if not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", "")), key


def location_text(axes) -> str:
    """How the agent expects a pointed-at voxel (``web_host_prompt``)."""
    return "[location axis0={}, axis1={}, axis2={}]".format(*[int(v) for v in axes])


def installed_agent_version():
    """Version of the nninteractive-agent installed in Slicer's Python, or None."""
    import importlib.metadata

    try:
        return importlib.metadata.version("nninteractive-agent")
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_agent_url():
    """Where pip installed nninteractive-agent from (direct_url.json): an archive URL or a local folder, or None."""
    import importlib.metadata

    try:
        return json.loads(importlib.metadata.distribution("nninteractive-agent").read_text("direct_url.json") or
                          "{}").get("url")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def latest_agent_commit() -> str:
    """The latest commit of the nninteractive-agent master branch."""
    import urllib.request

    request = urllib.request.Request(f"https://api.github.com/repos/{AGENT_REPOSITORY}/commits/master",
                                     headers={"Accept": "application/vnd.github.sha"})
    with urllib.request.urlopen(request, timeout=30) as response:
        commit = response.read().decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"unexpected answer: {commit[:100]}")
    return commit


def _module_available(name: str) -> bool:
    """``name`` can be imported. Checks a submodule, not just the package: nninteractive-client provides the
    nnInteractive package too, without local inference."""
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:  # a parent package is missing
        return False


def hex_to_rgb(color: str):
    color = color.lstrip("#")
    return [int(color[k:k + 2], 16) / 255.0 for k in (0, 2, 4)]

###############################################################################
# AgentConnection: login, websocket and HTTP requests (runs network IO in threads)
###############################################################################


class RequestError(RuntimeError):
    """An HTTP error answer of the server (``status``: its status code)."""

    def __init__(self, text: str, status: int):
        RuntimeError.__init__(self, text)
        self.status = status


class EndpointNotFoundError(RuntimeError):
    """The server has no such endpoint (an older nninteractive-agent)."""


class AgentConnection:
    """One logged-in session of the agent server.

    Everything that arrives (JSON events, label patches, results of background requests) is put in
    ``self.inbox`` by worker threads; the widget drains it on the main thread with ``poll``.
    The session cookie is kept, so reconnecting continues the same server session (and its
    nnInteractive seat) instead of starting another one.
    """

    def __init__(self):
        self.base = None
        self.verify = True
        self.cookie = None
        self.password = None
        self.inbox = queue.Queue()
        self._ws = None
        self._ws_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.connected = False

    # ------------------------------------------------------------- HTTP
    def url(self, path: str) -> str:
        return urllib.parse.urljoin(self.base, path)

    def _request(self, method, path, **kwargs):
        import requests

        kwargs.setdefault("timeout", 60)
        r = requests.request(method, self.url(path), cookies={COOKIE: self.cookie} if self.cookie else None,
                             verify=self.verify, **kwargs)
        if r.status_code == 401:
            raise PermissionError("not logged in")
        if r.status_code >= 400:
            raise RequestError(_detail(r), r.status_code)
        return r

    def login(self, base: str, password: str, verify: bool) -> None:
        """Blocking. Reuses ``self.cookie`` if the server still accepts it."""
        import requests

        if not verify:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        if base != self.base:
            self.cookie = None
        self.base, self.verify, self.password = base, verify, password
        if self.cookie:
            r = requests.get(self.url(""), cookies={COOKIE: self.cookie}, verify=verify, allow_redirects=False,
                             timeout=30)
            if r.status_code == 200:
                return
            self.cookie = None
        r = requests.post(self.url("login"), data={"password": password or ""}, verify=verify,
                          allow_redirects=False, timeout=30)
        cookie = r.cookies.get(COOKIE)
        if cookie is None:
            location = r.headers.get("location", "")
            if "locked" in location:
                raise PermissionError("Too many wrong passwords. Wait a few minutes and try again.")
            if "error" in location:
                raise PermissionError("Wrong password or key.")
            raise RuntimeError(f"{self.base} does not look like an nnInteractive agent server (HTTP {r.status_code}).")
        self.cookie = cookie

    def logout(self) -> None:
        if self.cookie:
            try:
                self._request("POST", "logout", allow_redirects=False, timeout=10)
            except Exception:  # noqa: BLE001 - best effort, the server forgets idle sessions anyway
                pass
        self.cookie = None

    def run_async(self, func, on_done=None) -> None:
        """Run ``func()`` in a thread; ``on_done(result, error)`` is called on the main thread."""

        def work():
            try:
                result, error = func(), None
            except Exception as e:  # noqa: BLE001 - reported to the caller
                result, error = None, e
            if on_done is not None:
                self.inbox.put(("call", on_done, result, error))

        threading.Thread(target=work, daemon=True).start()

    def upload_objects(self, path: str, segments: list, mode: str) -> None:
        """A label map on the agent's image grid becomes the agent's objects (``mode``: replace or add)."""
        with open(path, "rb") as f:
            try:
                self._request("POST", "api/objects",
                              files={"file": (os.path.basename(path), f, "application/octet-stream")},
                              data={"segments": json.dumps(segments), "mode": mode}, timeout=None)
            except RequestError as e:
                if e.status in (404, 405):  # a server without segmentation upload (it never answers 404 itself)
                    raise EndpointNotFoundError("api/objects") from e
                raise

    def upload(self, path: str, name: str) -> None:
        with open(path, "rb") as f:
            self._request("POST", "api/upload", files={"file": (name, f, "application/octet-stream")}, timeout=None)

    def download(self, path: str, dest: str) -> str:
        r = self._request("GET", path, timeout=None, stream=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(4 * 1024 * 1024):
                f.write(chunk)
        return dest

    # -------------------------------------------------------- websocket
    def start(self) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001, S110 - already closed
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.connected = False

    def send(self, cmd: dict) -> bool:
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            return False
        try:
            ws.send(json.dumps(cmd))
            return True
        except Exception:  # noqa: BLE001 - the receiving thread notices and reconnects
            return False

    def _run(self) -> None:
        import ssl

        import websocket

        url = re.sub(r"^http", "ws", self.url("api/ws"))
        retry = 0
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(
                    url, header=[f"Cookie: {COOKIE}={self.cookie}"], suppress_origin=True, timeout=30,
                    sslopt=None if self.verify else {"cert_reqs": ssl.CERT_NONE, "check_hostname": False},
                )
            except websocket.WebSocketBadStatusException as e:
                if e.status_code in (401, 403) and self.password is not None and retry == 0:
                    try:  # the session cookie expired: log in again once
                        self.cookie = None
                        self.login(self.base, self.password, self.verify)
                        retry += 1
                        continue
                    except Exception as e2:  # noqa: BLE001 - reported below
                        e = e2
                self.inbox.put(("closed", f"The server refused the connection: {e}", True))
                return
            except Exception as e:  # noqa: BLE001 - not reachable now: retry
                self.inbox.put(("closed", f"Cannot reach {self.base}: {e}", False))
                if self._stop.wait(min(10.0, 0.6 * 2**retry)):
                    return
                retry += 1
                continue
            ws.settimeout(None)
            with self._ws_lock:
                if self._stop.is_set():
                    ws.close()
                    return
                self._ws = ws
            retry = 0
            self.connected = True
            self.inbox.put(("open",))
            try:
                while not self._stop.is_set():
                    opcode, data = ws.recv_data()
                    if opcode == websocket.ABNF.OPCODE_TEXT:
                        ev = json.loads(data.decode())
                        if ev.get("t") == "result" and ev.get("images"):
                            ev["image_data"] = self._fetch_images(ev["images"][:MAX_CHAT_IMAGES])
                        self.inbox.put(("event", ev))
                    elif opcode == websocket.ABNF.OPCODE_BINARY:
                        self.inbox.put(("patch",) + decode_patch(data))
                    elif opcode == websocket.ABNF.OPCODE_CLOSE:
                        break
            except Exception as e:  # noqa: BLE001 - connection lost
                if not self._stop.is_set():
                    logging.info(f"nnInteractiveAgent: connection lost: {e}")
            finally:
                self.connected = False
                with self._ws_lock:
                    if self._ws is ws:
                        self._ws = None
            if not self._stop.is_set():
                self.inbox.put(("closed", "Connection lost, reconnecting ...", False))
                self._stop.wait(1.0)

    def _fetch_images(self, urls) -> list:
        out = []
        for u in urls:
            try:
                out.append(self._request("GET", u, timeout=30).content)
            except Exception:  # noqa: BLE001 - a missing thumbnail is cosmetic
                out.append(None)
        return out


###############################################################################
# InternalServer: nninteractive-agent-web run by Slicer's Python on this computer
###############################################################################


class InternalServer:
    """``nninteractive-agent-web`` in a child process, for this computer only (plain HTTP on 127.0.0.1;
    the password is passed in the environment). Its output lines are collected by a thread and read
    with ``read_log``."""

    def __init__(self):
        self.process = None
        self.port = None
        self.key = None
        self._log = queue.Queue()
        self._pump_thread = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    @staticmethod
    def port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def command(self, port: int, args: list) -> list:
        return [sys.executable, "-m", "nninteractive_agent.web.server", "--host", "127.0.0.1", "--port", str(port),
                *args]

    def start(self, port: int, key: str, args: list) -> None:
        self.stop()
        env = dict(os.environ, NNINTERACTIVE_WEB_PASSWORD=key, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True  # stop() ends the agent processes it starts as well
        cmd = self.command(port, args)
        self._log.put("$ " + " ".join(cmd))
        self.process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=env, **kwargs)
        self.port, self.key = port, key
        self._pump_thread = threading.Thread(target=self._pump, args=(self.process,), daemon=True)
        self._pump_thread.start()

    def _pump(self, process) -> None:
        for raw in iter(process.stdout.readline, b""):
            self._log.put(raw.decode("utf-8", errors="replace").rstrip())
        process.stdout.close()
        self._log.put(f"[the server stopped, exit code {process.wait()}]")

    def read_log(self, wait_for_exit: bool = False) -> list:
        """The output lines since the last call (``wait_for_exit``: all of them, after the process ended)."""
        if wait_for_exit and self._pump_thread is not None:
            self._pump_thread.join(timeout=3)
        lines = []
        while True:
            try:
                lines.append(self._log.get_nowait())
            except queue.Empty:
                return lines

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        try:
            if sys.platform == "win32":  # the whole tree: the agent CLIs the server started too
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001 - make sure it ends
            process.kill()

    def ready(self) -> bool:
        """The server answers HTTP (it starts listening after getting the weights and checking the LLM)."""
        import requests

        try:
            return requests.get(self.base + "login", timeout=1).status_code < 500
        except Exception:  # noqa: BLE001 - not listening yet
            return False


def _detail(response) -> str:
    try:
        return str(response.json().get("detail") or response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


###############################################################################
# LabelMirror: the server's label layers as segmentation nodes
###############################################################################


class LabelMirror:
    """Keeps numpy copies of the server's label layers ("objects", "current", "prompts") and writes
    the changed parts into two segmentation nodes on ``flush``: committed objects (one segment per
    object id) and the work in progress (the open object and its positive and negative prompts)."""

    def __init__(self):
        self.layers = {}  # name -> array in the volume's [k, j, i] grid
        self.objects = []  # rows of the last "objects" message
        self.current = None
        self.current_color = "#ffd400"
        self.volume_id = None
        self.image_tag = None  # "<key>:<n>" of the server image
        self.result_id = None
        self.work_id = None
        self.show3d = True
        self._dirty_ids = set()  # object ids whose segment must be rewritten
        self._dirty_work = False
        self._dirty_rows = False
        self._foreign = set()  # object segment IDs in the result node that this mirror did not write

    def reset(self) -> None:
        """A new image on the server: its layers come as full patches."""
        self.layers, self.objects, self.current = {}, [], None
        self.volume_id = self.image_tag = self.result_id = self.work_id = None
        self._dirty_ids, self._dirty_work, self._dirty_rows = set(), False, False
        self._foreign = set()

    @property
    def dirty(self) -> bool:
        return bool(self._dirty_ids or self._dirty_work or self._dirty_rows)

    # ------------------------------------------------------------ input
    def apply_patch(self, header: dict, crop) -> None:
        name, shape = header["layer"], tuple(header["shape"])
        sl = tuple(slice(lo, hi) for lo, hi in header["box"])
        arr = self.layers.get(name)
        full = header.get("full") or arr is None or arr.shape != shape or arr.dtype != np.dtype(header["dtype"])
        if name == "objects":
            if full:
                old_ids = np.unique(arr) if arr is not None else []
                self._dirty_ids.update(int(i) for i in old_ids)
                self._dirty_ids.update(int(r["id"]) for r in self.objects)
            else:
                self._dirty_ids.update(int(i) for i in np.unique(arr[sl]))
                self._dirty_ids.update(int(i) for i in np.unique(crop))
        else:
            self._dirty_work = True
        if full:
            arr = np.zeros(shape, np.dtype(header["dtype"]))
            self.layers[name] = arr
        arr[sl] = crop
        self._dirty_ids.discard(0)

    def set_objects(self, rows: list, current, current_color: str) -> None:
        self.objects, self.current, self.current_color = rows, current, current_color or self.current_color
        self._dirty_rows = self._dirty_work = True

    def bind(self, volume_node, image_tag: str) -> None:
        """Mirror into segmentation nodes of ``volume_node`` (found again by ``image_tag`` after a
        reconnect or a module reload)."""
        self.volume_id, self.image_tag = volume_node.GetID(), image_tag
        self.result_id = self.work_id = None
        self._dirty_ids.update(int(r["id"]) for r in self.objects)
        if "objects" in self.layers:
            self._dirty_ids.update(int(i) for i in np.unique(self.layers["objects"]))
        self._dirty_ids.discard(0)
        self._dirty_rows = self._dirty_work = True

    # ----------------------------------------------------------- output
    def volume_node(self):
        return slicer.mrmlScene.GetNodeByID(self.volume_id) if self.volume_id else None

    def _segmentation_node(self, role: str):
        volume = self.volume_node()
        if volume is None:
            return None
        attr = "result_id" if role == "result" else "work_id"
        node = slicer.mrmlScene.GetNodeByID(getattr(self, attr)) if getattr(self, attr) else None
        if node is None:
            for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                if n.GetAttribute(IMAGE_ATTRIBUTE) == self.image_tag and n.GetAttribute(ROLE_ATTRIBUTE) == role:
                    node = n
                    break
        if node is None:
            name = volume.GetName() + (" agent" if role == "result" else " agent work")
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", slicer.mrmlScene.GenerateUniqueName(name))
            node.SetAttribute(IMAGE_ATTRIBUTE, self.image_tag)
            node.SetAttribute(ROLE_ATTRIBUTE, role)
            node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
            node.SetNodeReferenceID(node.GetReferenceImageGeometryReferenceRole(), volume.GetID())
            node.CreateDefaultDisplayNodes()
            if role == "work":
                node.GetDisplayNode().SetOpacity2DFill(0.35)
            self._apply_3d(node, role)
        node.SetAndObserveTransformNodeID(volume.GetTransformNodeID())
        setattr(self, attr, node.GetID())
        return node

    def output_node(self, role: str):
        """The node ``role`` ("result" or "work") is mirrored into, None before the first flush."""
        node_id = self.result_id if role == "result" else self.work_id
        return slicer.mrmlScene.GetNodeByID(node_id) if node_id else None

    def set_output(self, role: str, node) -> None:
        """Mirror ``role`` into ``node`` from now on (None: a new node). The node that had this role keeps
        its segments but is no longer updated; the agent's segments in ``node`` are rewritten, other
        segments in it are kept."""
        for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            if n is not node and n.GetAttribute(IMAGE_ATTRIBUTE) == self.image_tag \
                    and n.GetAttribute(ROLE_ATTRIBUTE) == role:
                n.RemoveAttribute(IMAGE_ATTRIBUTE)
                n.RemoveAttribute(ROLE_ATTRIBUTE)
        if role == "result" and node is not None and node is not self.output_node("result"):
            # Object segments that were there before (e.g. from an earlier session) are not this mirror's:
            # they are kept unless an object of the agent takes over their ID.
            seg = node.GetSegmentation()
            self._foreign = {seg.GetNthSegmentID(k) for k in range(seg.GetNumberOfSegments())
                             if seg.GetNthSegmentID(k).startswith(OBJECT_SEGMENT_PREFIX)}  # fmt: skip
        if node is not None:
            node.SetAttribute(IMAGE_ATTRIBUTE, self.image_tag)
            node.SetAttribute(ROLE_ATTRIBUTE, role)
            if node.GetDisplayNode() is None:
                node.CreateDefaultDisplayNodes()
            self._apply_3d(node, role)
        setattr(self, "result_id" if role == "result" else "work_id", node.GetID() if node is not None else None)
        if role == "result":
            self._dirty_ids.update(int(r["id"]) for r in self.objects)
            self._dirty_rows = True
        else:
            self._dirty_work = True

    def set_show3d(self, show: bool) -> None:
        self.show3d = show
        for role in ("result", "work"):
            node = self._segmentation_node(role)
            if node is not None:
                self._apply_3d(node, role)

    def _apply_3d(self, node, role: str) -> None:
        if self.show3d:
            node.CreateClosedSurfaceRepresentation()
        else:
            node.RemoveClosedSurfaceRepresentation()
        if role == "work":  # prompts are lines in one slice: nothing to see in 3D
            for seg in (POS_SEGMENT, NEG_SEGMENT):
                if node.GetSegmentation().GetSegment(seg) is not None:
                    node.GetDisplayNode().SetSegmentVisibility3D(seg, False)

    def flush(self) -> bool:
        """Write what changed into the segmentation nodes. False if there is no volume to show it on."""
        volume = self.volume_node()
        if volume is None:
            return False
        shape = tuple(volume.GetImageData().GetDimensions())[::-1] if volume.GetImageData() else None
        layer_shape = next((a.shape for a in self.layers.values()), shape)
        if shape != layer_shape:
            logging.warning(f"nnInteractiveAgent: label layers {layer_shape} do not match the volume {shape}")
            return False
        result = self._segmentation_node("result")
        work = self._segmentation_node("work")
        rows = {int(r["id"]): r for r in self.objects}
        if self._dirty_rows:
            seg = result.GetSegmentation()
            for k in range(seg.GetNumberOfSegments() - 1, -1, -1):
                sid = seg.GetNthSegmentID(k)
                if sid.startswith(OBJECT_SEGMENT_PREFIX) and sid not in self._foreign \
                        and int(sid[len(OBJECT_SEGMENT_PREFIX):]) not in rows:
                    seg.RemoveSegment(sid)
            for i, r in rows.items():
                s = seg.GetSegment(OBJECT_SEGMENT_PREFIX + str(i))
                if s is None or OBJECT_SEGMENT_PREFIX + str(i) in self._foreign:
                    self._dirty_ids.add(i)
                else:
                    s.SetName(r["name"])
                    s.SetColor(hex_to_rgb(r["color"]))
                    _set_notes(s, r.get("notes", ""))
        objects = self.layers.get("objects")
        for i in sorted(self._dirty_ids):
            sid = OBJECT_SEGMENT_PREFIX + str(i)
            r = rows.get(i)
            if r is None:
                if result.GetSegmentation().GetSegment(sid) is not None and sid not in self._foreign:
                    result.GetSegmentation().RemoveSegment(sid)
                continue
            self._foreign.discard(sid)
            mask = (objects == i) if objects is not None else None
            _write_segment(result, sid, r["name"], r["color"], mask, volume)
            _set_notes(result.GetSegmentation().GetSegment(sid), r.get("notes", ""))
        if self._dirty_ids and self.show3d:  # an empty segmentation does not keep a surface representation
            self._apply_3d(result, "result")
        if self._dirty_work:
            cur = self.layers.get("current")
            name = f"Working: {self.current['name']}" if self.current else "Working"
            _write_segment(work, CURRENT_SEGMENT, name, self.current_color,
                           cur > 0 if (cur is not None and self.current) else None, volume)
            prompts = self.layers.get("prompts")
            for sid, label, value, color in ((POS_SEGMENT, "Positive prompts", PROMPT_POS, POS_COLOR),
                                             (NEG_SEGMENT, "Negative prompts", PROMPT_NEG, NEG_COLOR)):
                _write_segment(work, sid, label, color, prompts == value if prompts is not None else None, volume)
            self._apply_3d(work, "work")
        self._dirty_ids, self._dirty_work, self._dirty_rows = set(), False, False
        return True


def _write_segment(seg_node, segment_id, name, color, mask, volume) -> None:
    """Segment ``segment_id`` = ``mask`` (removed when the mask is None or empty)."""
    seg = seg_node.GetSegmentation()
    if mask is None or not mask.any():
        if seg.GetSegment(segment_id) is not None:
            seg.RemoveSegment(segment_id)
        return
    if seg.GetSegment(segment_id) is None:
        seg.AddEmptySegment(segment_id, name, hex_to_rgb(color))
    s = seg.GetSegment(segment_id)
    s.SetName(name)
    s.SetColor(hex_to_rgb(color))
    slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), seg_node, segment_id, volume)


def _set_notes(segment, notes: str) -> None:
    if segment is not None:
        segment.SetTag(NOTES_TAG, notes or "")


def _get_notes(segment) -> str:
    value = vtk.reference("")
    return str(value) if segment.GetTag(NOTES_TAG, value) else ""


class EnterKeyFilter(qt.QObject):
    """Calls ``on_enter(modifiers)`` for Enter in a text edit; the key is consumed if it returns True."""

    def __init__(self, on_enter):
        qt.QObject.__init__(self)
        self.on_enter = on_enter

    def eventFilter(self, _obj, event):
        if event.type() == qt.QEvent.KeyPress and event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            return bool(self.on_enter(int(event.modifiers())))
        return False


###############################################################################
# nnInteractiveAgentWidget
###############################################################################


class nnInteractiveAgentWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.conn = AgentConnection()
        self.mirror = LabelMirror()
        self.image_meta = None  # the server's current image
        self.bound_keys = {}  # server volume key -> volume node ID (images this Slicer sent or loaded)
        self.pending_upload = None  # volume node ID being uploaded
        self.replaying = False
        self.busy = False
        self.ready = False
        self.agent = None
        self._chat_images = 0
        self._downloading = None
        self._download_failed = {}  # image tag -> message, for downloads that failed (not retried automatically)
        self._last_flush = 0.0
        self._updating_selectors = False
        self.internal = InternalServer()
        self._connect_when_ready = False  # connect once the internal server answers
        self._next_ready_check = 0.0
        self._server_was_running = False
        self._run_log = []  # the last output lines of the internal server's current run
        self._sending_segmentation = False
        self._after_image = None  # action to run once the volume being sent is open (send a message or a segmentation)
        self._preferred_outputs = {}  # volume node ID -> {role: segmentation node ID}, chosen before it is sent
        self._logging_in = False
        self._after_connect = None  # action to run once connected (Send to agent or Send while disconnected)

    # ------------------------------------------------------------------ UI
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        # The tabs fill the module panel and the conversation takes the remaining height, so the chat and
        # the Send button fit without scrolling the panel.
        self.tabs = qt.QTabWidget()
        self.layout.addWidget(self.tabs, 1)
        agentTab = qt.QWidget()
        agentLayout = qt.QVBoxLayout(agentTab)
        agentLayout.setContentsMargins(4, 4, 4, 4)
        self.tabs.addTab(agentTab, _("Agent"))
        configScroll = qt.QScrollArea()  # its own scrolling, so it does not make the Agent tab taller
        configScroll.widgetResizable = True
        configScroll.frameShape = qt.QFrame.NoFrame
        configTab = qt.QWidget()
        configScroll.setWidget(configTab)
        self.tabs.addTab(configScroll, _("Configuration"))
        self.setupConfiguration(configTab)

        # Image
        box = ctk.ctkCollapsibleButton()
        box.text = _("Image")
        agentLayout.addWidget(box)
        form = qt.QFormLayout(box)
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        row = qt.QHBoxLayout()
        row.addWidget(self.volumeSelector, 1)
        self.uploadButton = qt.QPushButton(_("Send to agent"))
        self.uploadButton.toolTip = _("Upload this volume to the server; the agent then works on it")
        row.addWidget(self.uploadButton)
        form.addRow(_("Volume:"), row)
        self.outputSelectors = {}
        for role, label, tip in (
                ("result", _("Segmentation:"), _("Segmentation the agent's committed objects are written into")),
                ("work", _("Working segmentation:"), _("Segmentation the object the agent is working on and its "
                                                       "prompts are written into"))):
            selector = slicer.qMRMLNodeComboBox()
            selector.nodeTypes = ["vtkMRMLSegmentationNode"]
            selector.addEnabled = False
            selector.removeEnabled = False
            selector.renameEnabled = True
            selector.noneEnabled = True
            selector.noneDisplay = _("(create new)")
            selector.toolTip = tip + _(". Selecting another segmentation writes there from now on; the previous one "
                                       "keeps its segments.")
            selector.setMRMLScene(slicer.mrmlScene)
            self.outputSelectors[role] = selector
            if role == "work":  # rarely changed: it is in the Configuration tab
                self.conversationForm.addRow(label, selector)
                continue
            # A push button with a dropdown part (as Show 3D): the button replaces, the menu offers adding.
            self.sendSegmentationButton = ctk.ctkMenuButton()
            self.sendSegmentationButton.text = _("Send to agent")
            self.sendSegmentationButton.toolTip = _("Make this segmentation the agent's objects, to edit or discuss "
                                                    "them. If the agent works on another image than the selected "
                                                    "volume, the volume is sent first. (The arrow: add only the "
                                                    "segments the agent does not have.)")
            menu = qt.QMenu(self.sendSegmentationButton)
            menu.addAction(_("Replace the agent's objects")).connect("triggered()", lambda: self.onSendSegmentation("replace"))
            menu.addAction(_("Add new segments to the agent's objects")).connect(
                "triggered()", lambda: self.onSendSegmentation("add"))
            self.sendSegmentationButton.setMenu(menu)
            row = qt.QHBoxLayout()
            row.addWidget(selector, 1)
            row.addWidget(self.sendSegmentationButton)
            form.addRow(label, row)
            if role == "result":
                self.show3dButton = slicer.qMRMLSegmentationShow3DButton()
                self.show3dButton.toolTip = _("Show the segmentation and the working segmentation in 3D")
                self.show3dButton.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
                form.addRow(self.show3dButton)  # full width, as the segment list below it
                self.segmentsTable = slicer.qMRMLSegmentsTableView()
                self.segmentsTable.setMRMLScene(slicer.mrmlScene)
                self.segmentsTable.statusColumnVisible = False
                self.segmentsTable.jumpToSelectedSegmentEnabled = True
                # The module panel lays out by size hints: ignoring this one keeps the chat visible without
                # scrolling (the table then shows about three segments; it scrolls for more).
                self.segmentsTable.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Ignored)
                self.segmentsTable.minimumHeight = 80
                form.addRow(self.segmentsTable)

        # Chat
        box = ctk.ctkCollapsibleButton()
        box.text = _("Conversation")
        agentLayout.addWidget(box, 1)
        layout = qt.QVBoxLayout(box)
        # A green tint of the theme's button color for the Send button and the message box's border (works in
        # light and dark themes, keeps the native look of the button).
        base = slicer.app.palette().color(qt.QPalette.Button)
        tint = qt.QColor(int(base.red() * 0.75 + 46 * 0.25), int(base.green() * 0.75 + 204 * 0.25),
                         int(base.blue() * 0.75 + 64 * 0.25))
        self.chat = qt.QTextBrowser()
        self.chat.openLinks = False
        self.chat.minimumHeight = 80
        self.chat.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Ignored)  # takes the remaining height
        layout.addWidget(self.chat, 1)
        self.statusLabel = qt.QLabel(_("Not connected. Sending a volume or a message connects to the agent server "
                                       "set up in the Configuration tab."))
        self.statusLabel.wordWrap = True
        layout.addWidget(self.statusLabel)
        self.messageEdit = qt.QPlainTextEdit()
        self.messageEdit.maximumHeight = 70
        self.messageEdit.setStyleSheet(f"QPlainTextEdit {{ border: 2px solid {tint.name()}; border-radius: 3px; }}")
        layout.addWidget(self.messageEdit)
        row = qt.QHBoxLayout()
        self.sendButton = qt.QPushButton(_("Send"))
        palette = self.sendButton.palette
        palette.setColor(qt.QPalette.Button, tint)
        self.sendButton.palette = palette
        self.stopButton = qt.QPushButton()
        self.stopButton.icon = qt.QIcon(self.resourcePath("Icons/stop.svg"))
        self.stopButton.toolTip = _("Stop the agent (interrupts its current turn; committed objects and the "
                                    "conversation are kept)")
        self.stopButton.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Preferred)
        self.stopButton.setFixedHeight(self.sendButton.sizeHint.height())  # icon-only buttons come out taller
        self.locationButton = qt.QPushButton(_("Insert"))
        self.locationButton.toolTip = _("Point the agent at places: insert the crosshair position, or a point "
                                        "list, line, curve or ROI (sent as voxel coordinates)")
        self.insertMenu = qt.QMenu(self.locationButton)
        self.locationButton.setMenu(self.insertMenu)
        self.newConversationButton = qt.QPushButton(_("New conversation"))
        self.newConversationButton.toolTip = _("Start a new conversation; the objects are kept")
        self.followButton = qt.QPushButton(_("Views follow agent"))
        self.followButton.checkable = True
        self.followButton.checked = True
        self.followButton.toolTip = _("Move the slice views to where the agent prompts or looks")
        for b in (self.sendButton, self.stopButton, self.locationButton, self.newConversationButton,
                  self.followButton):
            row.addWidget(b)
        layout.addLayout(row)

        self.uploadButton.connect("clicked()", self.onUpload)
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda _node: self.updateGUI())
        self.sendSegmentationButton.connect("clicked()", lambda: self.onSendSegmentation("replace"))
        for role, selector in self.outputSelectors.items():
            selector.connect("currentNodeChanged(vtkMRMLNode*)", lambda node, role=role: self.onOutputSelected(role, node))
        # The button switches the 3D surfaces of the segmentation; the working segmentation follows it.
        self.show3dButton.connect("toggled(bool)", self.mirror.set_show3d)
        self.modelCombo.connect("activated(int)", self.onSettingsChanged)
        self.effortCombo.connect("activated(int)", self.onSettingsChanged)
        self.chat.connect("anchorClicked(QUrl)", self.onChatLink)
        self.sendButton.connect("clicked()", self.onSend)
        self.stopButton.connect("clicked()", lambda: self.conn.send({"cmd": "stop"}))
        self.insertMenu.connect("aboutToShow()", self.onInsertMenu)
        self.newConversationButton.connect("clicked()", lambda: self.conn.send({"cmd": "new_conversation"}))
        self.enterKeyFilter = EnterKeyFilter(self.onEnterKey)
        self.messageEdit.installEventFilter(self.enterKeyFilter)
        self.onEnterSendsChanged()

        self.timer = qt.QTimer()
        self.timer.setInterval(100)
        self.timer.connect("timeout()", self.poll)
        self.timer.start()
        self.updateGUI()

    def setupConfiguration(self, tab) -> None:
        settings = qt.QSettings()
        layout = qt.QVBoxLayout(tab)

        box = ctk.ctkCollapsibleButton()
        box.text = _("nninteractive-agent version")
        box.collapsed = True
        layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.installedVersionLabel = qt.QLabel("")
        self.installedVersionLabel.wordWrap = True
        self.installedVersionLabel.textInteractionFlags = qt.Qt.TextSelectableByMouse
        form.addRow(_("Installed:"), self.installedVersionLabel)
        row = qt.QHBoxLayout()
        self.installRecommendedButton = qt.QPushButton(_("Install recommended version"))
        self.installRecommendedButton.toolTip = _("Install the version this module was tested with (commit {commit}) "
                                                  "in Slicer's Python; stops the internal server").format(
                                                      commit=AGENT_COMMIT[:7])
        self.installLatestButton = qt.QPushButton(_("Install latest version"))
        self.installLatestButton.toolTip = _("Install the latest commit of the nninteractive-agent master branch "
                                             "in Slicer's Python; stops the internal server")
        row.addWidget(self.installRecommendedButton)
        row.addWidget(self.installLatestButton)
        form.addRow(row)
        row = qt.QHBoxLayout()
        package = settings.value(SETTINGS_PREFIX + "Package", "")
        if package.endswith("/archive/refs/heads/master.zip"):  # the default of earlier versions of this module
            package = ""
        self.packageEdit = qt.QLineEdit(package)
        self.packageEdit.placeholderText = _("other pip requirement, e.g. C:/src/nninteractive-agent[web]")
        self.packageEdit.toolTip = _("Install nninteractive-agent from another source, such as a local checkout")
        self.installCustomButton = qt.QPushButton(_("Install"))
        row.addWidget(self.packageEdit, 1)
        row.addWidget(self.installCustomButton)
        form.addRow(_("Other:"), row)
        self.updateInstalledVersion()

        self.internalRadio = qt.QRadioButton(_("Internal server (Slicer installs and starts it on this computer)"))
        layout.addWidget(self.internalRadio)
        self.internalWidget = qt.QWidget()
        form = qt.QFormLayout(self.internalWidget)
        form.setContentsMargins(20, 0, 0, 0)
        self.llmCombo = qt.QComboBox()
        for name, label in LLM_BACKENDS:
            self.llmCombo.addItem(label, name)
        self.llmCombo.toolTip = _("The agent's LLM and how it signs in. A subscription must be logged in on this "
                                  "computer once (run claude, or codex login, in a terminal).")
        self.llmCombo.currentIndex = max(0, self.llmCombo.findData(settings.value(SETTINGS_PREFIX + "LLM", "claude")))
        form.addRow(_("Agent LLM:"), self.llmCombo)
        self.llmUrlEdit = qt.QLineEdit(settings.value(SETTINGS_PREFIX + "LLMUrl", ""))
        self.llmUrlEdit.placeholderText = "http://localhost:8000"
        self.llmUrlEdit.toolTip = _("The Anthropic-compatible server of the model")
        form.addRow(_("LLM server:"), self.llmUrlEdit)
        self.apiKeyFileEdit = ctk.ctkPathLineEdit()
        self.apiKeyFileEdit.filters = ctk.ctkPathLineEdit.Files
        self.apiKeyFileEdit.currentPath = settings.value(SETTINGS_PREFIX + "ApiKeyFile", "")
        self.apiKeyFileEdit.toolTip = _("File with the API key (else $ANTHROPIC_API_KEY, $OPENAI_API_KEY or "
                                        "$LLM_API_KEY is used)")
        form.addRow(_("API key file:"), self.apiKeyFileEdit)
        self.inferenceCombo = qt.QComboBox()
        self.inferenceCombo.addItem(_("This computer (needs an NVIDIA GPU)"), "local")
        self.inferenceCombo.addItem(_("nnInteractive server"), "server")
        self.inferenceCombo.currentIndex = max(0, self.inferenceCombo.findData(
            settings.value(SETTINGS_PREFIX + "Inference", "local")))
        form.addRow(_("nnInteractive runs on:"), self.inferenceCombo)
        self.inferenceUrlEdit = qt.QLineEdit(settings.value(SETTINGS_PREFIX + "InferenceUrl", ""))
        self.inferenceUrlEdit.placeholderText = "http://gpu-machine:1527"
        self.inferenceUrlEdit.toolTip = _("URL of an nninteractive-server")
        form.addRow(_("nnInteractive server:"), self.inferenceUrlEdit)
        self.portSpin = qt.QSpinBox()
        self.portSpin.setRange(1024, 65535)
        self.portSpin.value = int(settings.value(SETTINGS_PREFIX + "Port", DEFAULT_INTERNAL_PORT))
        form.addRow(_("Port:"), self.portSpin)
        self.startStopButton = qt.QPushButton(_("Start server"))
        form.addRow(self.startStopButton)
        layout.addWidget(self.internalWidget)

        self.remoteRadio = qt.QRadioButton(_("Remote server (nninteractive-agent-web started elsewhere)"))
        layout.addWidget(self.remoteRadio)
        self.remoteWidget = qt.QWidget()
        form = qt.QFormLayout(self.remoteWidget)
        form.setContentsMargins(20, 0, 0, 0)
        self.serverEdit = qt.QLineEdit(settings.value(SETTINGS_PREFIX + "Server", DEFAULT_SERVER))
        self.serverEdit.toolTip = _("The link nninteractive-agent-web prints at start-up "
                                    "(http://127.0.0.1:8443/?key=...), or the server's address")
        form.addRow(_("Server:"), self.serverEdit)
        self.passwordEdit = qt.QLineEdit()
        self.passwordEdit.echoMode = qt.QLineEdit.Password
        self.passwordEdit.placeholderText = _("not needed if the link contains ?key=")
        form.addRow(_("Password:"), self.passwordEdit)
        self.insecureCheck = qt.QCheckBox(_("Accept a self-signed certificate"))
        self.insecureCheck.checked = settings.value(SETTINGS_PREFIX + "Insecure", "false") == "true"
        form.addRow("", self.insecureCheck)
        layout.addWidget(self.remoteWidget)
        internal = settings.value(SETTINGS_PREFIX + "Mode", "internal") == "internal"
        self.internalRadio.checked = internal
        self.remoteRadio.checked = not internal

        row = qt.QHBoxLayout()
        self.connectButton = qt.QPushButton(_("Connect"))
        self.connectButton.toolTip = _("Also happens when you send a volume or a message")
        self.disconnectButton = qt.QPushButton(_("Disconnect"))
        self.endSessionButton = qt.QPushButton(_("End session"))
        self.endSessionButton.toolTip = _("Close the server session: frees its nnInteractive seat and forgets "
                                          "the conversation (the segmentation stays in Slicer)")
        for b in (self.connectButton, self.disconnectButton, self.endSessionButton):
            row.addWidget(b)
        layout.addLayout(row)
        self.serverInfoLabel = qt.QLabel("")
        self.serverInfoLabel.wordWrap = True
        layout.addWidget(self.serverInfoLabel)

        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel(_("Server output:")))
        row.addStretch(1)
        clearButton = qt.QPushButton(_("Clear"))
        row.addWidget(clearButton)
        layout.addLayout(row)
        self.serverLog = qt.QPlainTextEdit()
        self.serverLog.readOnly = True
        self.serverLog.maximumBlockCount = 5000
        self.serverLog.minimumHeight = 100
        layout.addWidget(self.serverLog, 1)

        box = ctk.ctkCollapsibleButton()
        box.text = _("Conversation")
        layout.addWidget(box)
        form = qt.QFormLayout(box)
        self.conversationForm = form  # setup() adds the working segmentation selector here
        self.modelCombo = qt.QComboBox()
        self.modelCombo.toolTip = _("The agent's model (the server lists them once connected)")
        form.addRow(_("Model:"), self.modelCombo)
        self.effortCombo = qt.QComboBox()
        self.effortCombo.toolTip = _("How much the model thinks: higher is more careful and slower")
        form.addRow(_("Effort:"), self.effortCombo)
        self.showImagesCheck = qt.QCheckBox(_("Show images in conversation window"))
        self.showImagesCheck.toolTip = _("Show thumbnails of the images the agent looked at in the conversation")
        self.showImagesCheck.checked = settings.value(SETTINGS_PREFIX + "ShowImages", "true") == "true"
        form.addRow(self.showImagesCheck)
        self.enterSendsCheck = qt.QCheckBox(_("Enter sends message"))
        self.enterSendsCheck.toolTip = _("On: Enter sends, Shift+Enter starts a new line, no Send button. "
                                         "Off: Ctrl+Enter or the Send button sends, Enter starts a new line.")
        self.enterSendsCheck.checked = settings.value(SETTINGS_PREFIX + "EnterSends", "true") == "true"
        form.addRow(self.enterSendsCheck)

        self.internalRadio.connect("toggled(bool)", self.onModeChanged)
        self.llmCombo.connect("currentIndexChanged(int)", lambda _i: self.updateConfigurationGUI())
        self.inferenceCombo.connect("currentIndexChanged(int)", lambda _i: self.updateConfigurationGUI())
        self.startStopButton.connect("clicked()", self.onStartStopServer)
        self.installRecommendedButton.connect("clicked()", lambda: self.onInstallAgent("recommended"))
        self.installLatestButton.connect("clicked()", lambda: self.onInstallAgent("latest"))
        self.installCustomButton.connect("clicked()", lambda: self.onInstallAgent("custom"))
        self.packageEdit.connect("textChanged(QString)", lambda _t: self.updateConfigurationGUI())
        self.connectButton.connect("clicked()", self.onConnect)
        self.disconnectButton.connect("clicked()", self.onDisconnect)
        self.endSessionButton.connect("clicked()", self.onEndSession)
        clearButton.connect("clicked()", self.serverLog.clear)
        self.showImagesCheck.connect("toggled(bool)", lambda checked: qt.QSettings().setValue(
            SETTINGS_PREFIX + "ShowImages", "true" if checked else "false"))
        self.enterSendsCheck.connect("toggled(bool)", self.onEnterSendsChanged)

    def cleanup(self) -> None:
        self.timer.stop()
        self.conn.stop()
        self.internal.stop()
        self.removeObservers()

    def connecting(self) -> bool:
        return self._connect_when_ready or self._logging_in or (self.conn._thread is not None and not self.conn.connected)

    def updateConfigurationGUI(self) -> None:
        internal = self.internalRadio.checked
        locked = self.conn.connected or self.connecting()  # settings apply to the next connection
        self.internalRadio.enabled = self.remoteRadio.enabled = not locked
        self.internalWidget.enabled = internal
        self.remoteWidget.enabled = not internal and not locked
        running = self.internal.running
        for w in (self.llmCombo, self.llmUrlEdit, self.apiKeyFileEdit, self.inferenceCombo, self.inferenceUrlEdit,
                  self.portSpin):
            w.enabled = not running and not self._connect_when_ready
        llm = self.llmCombo.currentData
        self.llmUrlEdit.enabled = self.llmUrlEdit.enabled and llm == "server"
        self.apiKeyFileEdit.enabled = self.apiKeyFileEdit.enabled and llm in ("claude-api", "codex-api", "server")
        self.inferenceUrlEdit.enabled = self.inferenceUrlEdit.enabled and self.inferenceCombo.currentData == "server"
        self.startStopButton.text = _("Stop server") if running else _("Start server")
        self.installCustomButton.enabled = bool(self.packageEdit.text.strip())

    def updateGUI(self) -> None:
        connected = self.conn.connected
        connecting = self.connecting()
        self.connectButton.enabled = not connected and not connecting
        self.disconnectButton.enabled = connected or connecting
        self.endSessionButton.enabled = self.conn.cookie is not None
        self.updateConfigurationGUI()
        idle = (connected and not self.busy and self.pending_upload is None and self._downloading is None
                and not self._sending_segmentation)
        can_connect = not connected and not connecting  # these buttons connect first
        self.uploadButton.enabled = idle or can_connect
        self.updateOutputSelectors()
        # Without an image or a connection, sending the segmentation connects and sends the volume first.
        self.sendSegmentationButton.enabled = ((idle or can_connect)
                                               and self.outputSelectors["result"].currentNode() is not None)
        self.sendButton.enabled = idle or can_connect  # without an image, sending explains what is missing
        self.stopButton.enabled = connected and self.busy
        self.newConversationButton.enabled = idle
        self.modelCombo.enabled = self.effortCombo.enabled = connected and not self.busy

    def outputsAreLive(self) -> bool:
        """The output selectors are the agent's current outputs: the selected volume is the agent's image."""
        volume = self.mirror.volume_node()
        return volume is not None and self.volumeSelector.currentNode() in (None, volume)

    def preferredOutputs(self) -> dict:
        """Segmentations chosen for the selected volume, used once it is sent to the agent (role -> node ID)."""
        volume = self.volumeSelector.currentNode()
        return self._preferred_outputs.setdefault(volume.GetID() if volume is not None else "", {})

    def updateOutputSelectors(self) -> None:
        """Show the nodes the mirror writes into; for a volume that is not the agent's image, the nodes the user
        chose for it."""
        live = self.outputsAreLive()
        self._updating_selectors = True
        try:
            for role, selector in self.outputSelectors.items():
                if live:
                    node = self.mirror.output_node(role)
                else:
                    node = slicer.mrmlScene.GetNodeByID(self.preferredOutputs().get(role) or "")
                if selector.currentNode() is not node:
                    selector.setCurrentNode(node)
            result = self.outputSelectors["result"].currentNode()
            if self.show3dButton.segmentationNode() is not result:
                self.show3dButton.setSegmentationNode(result)
                self.segmentsTable.setSegmentationNode(result)
        finally:
            self._updating_selectors = False

    # -------------------------------------------------------------- actions
    def saveConfiguration(self) -> None:
        settings = qt.QSettings()
        for key, value in (("Mode", "internal" if self.internalRadio.checked else "remote"),
                           ("LLM", self.llmCombo.currentData), ("LLMUrl", self.llmUrlEdit.text.strip()),
                           ("ApiKeyFile", self.apiKeyFileEdit.currentPath),
                           ("Inference", self.inferenceCombo.currentData),
                           ("InferenceUrl", self.inferenceUrlEdit.text.strip()), ("Port", self.portSpin.value),
                           ("Package", self.packageEdit.text.strip()), ("Server", self.serverEdit.text.strip()),
                           ("Insecure", "true" if self.insecureCheck.checked else "false")):
            settings.setValue(SETTINGS_PREFIX + key, value)

    def onModeChanged(self, _internal=None) -> None:
        self.saveConfiguration()
        self.updateGUI()

    def onConnect(self) -> None:
        if not self.installPackages(["websocket-client"], ["websocket"]):
            self._after_connect = None
            return
        self._download_failed.clear()
        self.saveConfiguration()
        if self.internalRadio.checked:
            if not self.internal.running and not self.startInternalServer():
                self._after_connect = None
                return
            self._connect_when_ready = True  # poll() logs in once the server answers
            self._next_ready_check = 0.0
            self.setStatus(_("Starting the agent server ... (Configuration tab: server output)"))
            self.updateGUI()
            return
        base, key = parse_server_link(self.serverEdit.text)
        self.login(base, key or self.passwordEdit.text, not self.insecureCheck.checked)

    def login(self, base: str, password: str, verify: bool) -> None:
        cookie = qt.QSettings().value(SETTINGS_PREFIX + "Cookie/" + base, "")
        if cookie and (self.conn.cookie is None or self.conn.base != base):
            self.conn.base, self.conn.cookie = base, cookie  # continue the last session with this server
        self.setStatus(_("Connecting to {base} ...").format(base=base))
        self._logging_in = True
        self.updateGUI()

        def done(_result, error):
            self._logging_in = False
            if error is not None:
                self._after_connect = None
                self.setStatus(_("Not connected"))
                slicer.util.errorDisplay(_("Could not log in to {base}: {error}").format(base=base, error=error))
                self.updateGUI()
                return
            qt.QSettings().setValue(SETTINGS_PREFIX + "Cookie/" + base, self.conn.cookie)
            self.conn.start()
            self.updateGUI()

        self.conn.run_async(lambda: self.conn.login(base, password, verify), done)

    def connectThen(self, action) -> None:
        """Connect (starting the internal server if needed), then run ``action`` once the session is replayed."""
        if self.connecting():
            return
        self._after_connect = action
        self.onConnect()

    def onDisconnect(self) -> None:
        self._connect_when_ready = False
        self._after_connect = None
        self.conn.stop()
        self.setStatus(_("Not connected"))
        self.updateGUI()

    # ------------------------------------------------------- internal server
    def installPackages(self, requirements: list, modules: list) -> bool:
        """pip install ``requirements`` unless all ``modules`` can be imported (asks first)."""
        if all(importlib.util.find_spec(m) is not None for m in modules):
            return True
        if not slicer.util.confirmOkCancelDisplay(
                _("The nnInteractive Agent module needs these Python packages: {packages}. Install them now?").format(
                    packages=", ".join(requirements))):
            return False
        return self._pip_install(requirements)

    def _pip_install(self, requirements: list, args=()) -> bool:
        dialog = slicer.util.createProgressDialog(
            labelText=_("Installing {packages} ... (this can take several minutes)").format(
                packages=", ".join(requirements)), maximum=0)
        slicer.app.processEvents()
        try:
            for requirement in requirements:
                slicer.util.pip_install([*args, requirement])
        except Exception as e:  # noqa: BLE001 - shown to the user
            slicer.util.errorDisplay(_("Could not install {packages}: {error}").format(
                packages=", ".join(requirements), error=e))
            return False
        finally:
            dialog.close()
        importlib.invalidate_caches()
        return True

    def installInternalServer(self, local: bool) -> bool:
        """nninteractive-agent with the web demo in Slicer's Python, and nnInteractive for local inference
        (PyTorch through the NNUNet extension, as the nnInteractive module does)."""
        if local and not _module_available("nnInteractive.inference.inference_session"):
            try:
                from SlicerNNUNetLib import InstallLogic
            except ImportError:
                slicer.util.errorDisplay(_("nnInteractive on this computer needs the NNUNet extension, which installs "
                                           "PyTorch. Install it in the Extensions Manager and restart Slicer, or "
                                           "select an nnInteractive server in the Configuration tab."))
                return False
            if not slicer.util.confirmOkCancelDisplay(_("nnInteractive on this computer needs PyTorch and nnInteractive "
                                                        "installed in Slicer's Python. Install them now? (this can "
                                                        "take several minutes)")):
                return False
            if not InstallLogic().setupPythonRequirements():
                slicer.util.errorDisplay(_("PyTorch and nnU-Net could not be installed."))
                return False
            if not self._pip_install([LOCAL_INFERENCE_PACKAGE]):
                return False
        if not self.installPackages([AGENT_PACKAGE], list(AGENT_MODULES)):
            return False
        # As TotalSegmentator does: offer the recommended version if another one is installed, unless the user
        # chose that one in the "nninteractive-agent version" section.
        installed = installed_agent_url()
        settings = qt.QSettings()
        if installed and installed != agent_archive_url(AGENT_COMMIT) and \
                settings.value(SETTINGS_PREFIX + "AgentVersion", "recommended") == "recommended" and \
                settings.value(SETTINGS_PREFIX + "DeclinedAgentCommit", "") != AGENT_COMMIT:
            box = qt.QMessageBox(slicer.util.mainWindow())
            box.icon = qt.QMessageBox.Question
            box.windowTitle = _("Update nninteractive-agent")
            box.text = _("This module recommends another version of nninteractive-agent than the installed one.")
            box.detailedText = _("Installed: {installed}\n\nRecommended: {recommended}\n\nKeep current version is "
                                 "not asked again for this recommended version. The nninteractive-agent version "
                                 "section of the Configuration tab installs any version.").format(
                                     installed=installed, recommended=agent_archive_url(AGENT_COMMIT))
            switchButton = box.addButton(_("Switch to recommended version"), qt.QMessageBox.AcceptRole)
            keepButton = box.addButton(_("Keep current version"), qt.QMessageBox.DestructiveRole)
            box.addButton(qt.QMessageBox.Cancel)  # also Esc and the window's close button
            box.setDefaultButton(switchButton)
            box.exec_()
            clicked = box.clickedButton()
            box.deleteLater()
            if clicked == switchButton:
                return self.installAgent(AGENT_PACKAGE)
            if clicked != keepButton:
                return False  # Cancel: not starting the server, and ask again next time
            settings.setValue(SETTINGS_PREFIX + "DeclinedAgentCommit", AGENT_COMMIT)
        return True

    def startInternalServer(self) -> bool:
        self.saveConfiguration()
        local = self.inferenceCombo.currentData == "local"
        inference_url = self.inferenceUrlEdit.text.strip()
        if not local and not inference_url:
            slicer.util.errorDisplay(_("Enter the URL of the nnInteractive server in the Configuration tab."))
            return False
        if not self.installInternalServer(local):
            return False
        port = self.portSpin.value
        if InternalServer.port_in_use(port):
            slicer.util.errorDisplay(_("Port {port} is in use (by another server?). Choose another port in the "
                                       "Configuration tab.").format(port=port))
            return False
        settings = qt.QSettings()
        key = settings.value(SETTINGS_PREFIX + "InternalKey", "")
        if not key:  # kept, so the session cookie stays valid when the server is restarted
            key = secrets.token_urlsafe(12)
            settings.setValue(SETTINGS_PREFIX + "InternalKey", key)
        llm = self.llmCombo.currentData
        args = ["--llm", llm, "--server", "local" if local else inference_url]
        if llm == "server" and self.llmUrlEdit.text.strip():
            args += ["--llm-url", self.llmUrlEdit.text.strip()]
        if llm in ("claude-api", "codex-api", "server") and self.apiKeyFileEdit.currentPath:
            args += ["--api-key-file", self.apiKeyFileEdit.currentPath]
        self._run_log = []
        try:
            self.internal.start(port, key, args)
        except OSError as e:
            slicer.util.errorDisplay(_("Could not start the agent server: {error}").format(error=e))
            return False
        self.updateGUI()
        return True

    def stopInternalServer(self) -> None:
        if self.conn.base == self.internal.base:
            self.onDisconnect()
        self._connect_when_ready = False
        self.internal.stop()
        self.updateGUI()

    def onStartStopServer(self) -> None:
        if self.internal.running:
            self.stopInternalServer()
        else:
            self.startInternalServer()

    def onInstallAgent(self, version: str) -> None:
        """Install nninteractive-agent: the ``recommended`` version, the ``latest`` master commit, or the
        ``custom`` requirement."""
        if version == "recommended":
            requirement = AGENT_PACKAGE
        elif version == "latest":
            try:
                commit = latest_agent_commit()
            except Exception as e:  # noqa: BLE001 - shown to the user
                slicer.util.errorDisplay(_("Could not get the latest version of nninteractive-agent: {error}").format(
                    error=e))
                return
            requirement = f"nninteractive-agent[web] @ {agent_archive_url(commit)}"
        else:
            requirement = self.packageEdit.text.strip()
        if not slicer.util.confirmOkCancelDisplay(
                _("Install nninteractive-agent from {package}? This stops the internal server.").format(
                    package=requirement), _("Install nninteractive-agent")):
            return
        self.stopInternalServer()
        if self.installAgent(requirement):
            qt.QSettings().setValue(SETTINGS_PREFIX + "AgentVersion", version)
            self.saveConfiguration()

    def installAgent(self, requirement: str) -> bool:
        # The requested code even if the version number is the same, then whatever it needs.
        ok = self._pip_install([requirement], ["--upgrade", "--force-reinstall", "--no-deps"]) and \
            self._pip_install([requirement])
        self.updateInstalledVersion()
        return ok

    def updateInstalledVersion(self) -> None:
        version, url = installed_agent_version(), installed_agent_url()
        if version is None:
            text = _("not installed")
        elif url == agent_archive_url(AGENT_COMMIT):
            text = _("{version}, recommended version (commit {commit})").format(version=version,
                                                                               commit=AGENT_COMMIT[:7])
        else:
            match = re.fullmatch(rf"https://github\.com/{AGENT_REPOSITORY}/archive/([0-9a-f]{{40}})\.zip", url or "")
            source = _("commit {commit}").format(commit=match.group(1)[:7]) if match else (url or _("unknown source"))
            text = _("{version}, from {source}").format(version=version, source=source)
        self.installedVersionLabel.text = text

    def pollInternalServer(self) -> None:
        lines = self.internal.read_log()
        if lines:
            self._run_log = (self._run_log + lines)[-12:]
            self.serverLog.appendPlainText("\n".join(lines))
        if not self._connect_when_ready or time.time() < self._next_ready_check:
            return
        self._next_ready_check = time.time() + 1.0
        if not self.internal.running:
            self._connect_when_ready = False
            self._after_connect = None
            lines = self.internal.read_log(wait_for_exit=True)
            if lines:
                self._run_log = (self._run_log + lines)[-12:]
                self.serverLog.appendPlainText("\n".join(lines))
            self.setStatus(_("The agent server stopped"))
            slicer.util.errorDisplay(_("The agent server stopped. Its last output:\n\n{output}").format(
                output="\n".join(self._run_log)))
            self.updateGUI()
        elif self.internal.ready():
            self._connect_when_ready = False
            self.login(self.internal.base, self.internal.key, True)

    def onEndSession(self) -> None:
        if not slicer.util.confirmOkCancelDisplay(_("End the session on the server? The conversation is forgotten; "
                                                    "the segmentation stays in Slicer.")):
            return
        base = self.conn.base
        self.conn.stop()
        self.conn.logout()
        if base:
            qt.QSettings().remove(SETTINGS_PREFIX + "Cookie/" + base)
        self.image_meta = None
        self.mirror.reset()
        self.setStatus(_("Session ended"))
        self.updateGUI()

    def onUpload(self) -> None:
        volume = self.volumeSelector.currentNode()
        if volume is None or volume.GetImageData() is None:
            slicer.util.errorDisplay(_("Select a volume first."))
            return
        if volume.GetImageData().GetNumberOfScalarComponents() != 1:
            slicer.util.errorDisplay(_("Only single-component (grayscale) volumes can be sent to the agent."))
            return
        if not self.conn.connected:
            self.connectThen(self.onUpload)
            return
        import SimpleITK as sitk
        import sitkUtils

        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", volume.GetName())[:100] or "volume"
        path = os.path.join(tempfile.mkdtemp(prefix="nnInteractiveAgent_"), name + ".nrrd")
        self.setStatus(_("Preparing {name} ...").format(name=volume.GetName()))
        slicer.app.processEvents()
        # The NRRD keeps the voxel grid, so the server's arrays have the shape and order of arrayFromVolume.
        sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(volume), path, True)
        self.pending_upload = volume.GetID()
        self.setStatus(_("Uploading {name} ...").format(name=volume.GetName()))
        self.updateGUI()

        def done(_result, error):
            if error is not None:
                self.pending_upload = None
                self._after_image = None
                slicer.util.errorDisplay(_("Could not send the volume: {error}").format(error=error))
            try:
                os.remove(path)
                os.rmdir(os.path.dirname(path))
            except OSError:
                pass
            self.updateGUI()

        self.conn.run_async(lambda: self.conn.upload(path, name + ".nrrd"), done)

    def onSendSegmentation(self, mode: str, node_id: str = None) -> None:
        """The selected segmentation becomes the agent's objects (``replace``), or its segments that are not
        the agent's are added (``add``). The objects then come back into it as the agent's segments.
        Connects and sends the selected volume first if needed."""
        segmentation = (slicer.mrmlScene.GetNodeByID(node_id) if node_id
                        else self.outputSelectors["result"].currentNode())
        if segmentation is None or not self.sendSegmentationButton.enabled:
            return
        again = lambda: self.onSendSegmentation(mode, segmentation.GetID())  # noqa: E731
        if not self.conn.connected:
            self.connectThen(again)
            return
        volume = self.mirror.volume_node()
        selected = self.volumeSelector.currentNode()
        if volume is None or (selected is not None and selected is not volume):
            # The agent has no image here, or works on another one: the selected volume goes first, then the
            # segmentation (which also becomes the output for it).
            if selected is None:
                slicer.util.errorDisplay(_("Select the volume of the segmentation, so it can be sent to the agent "
                                           "with it."))
                return
            self.preferredOutputs()["result"] = segmentation.GetID()
            self._after_image = again
            self.onUpload()
            return
        seg = segmentation.GetSegmentation()
        ids = [seg.GetNthSegmentID(k) for k in range(seg.GetNumberOfSegments())]
        if mode == "add":  # the agent's own segments are its objects already
            ids = [i for i in ids if not i.startswith(OBJECT_SEGMENT_PREFIX)]
        if not ids:
            slicer.util.errorDisplay(_("{name} has no segments to send.").format(name=segmentation.GetName())
                                     if mode == "replace" else _("All segments are the agent's objects already."))
            return
        import SimpleITK as sitk
        import sitkUtils

        # One label value per segment (index + 1) on the agent's image grid; a later segment wins where
        # segments overlap (every voxel belongs to one object).
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            id_array = vtk.vtkStringArray()
            for i in ids:
                id_array.InsertNextValue(i)
            if not slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentation, id_array, labelmap,
                                                                                    volume):
                slicer.util.errorDisplay(_("Could not export the segmentation on the agent's image grid."))
                return
            path = os.path.join(tempfile.mkdtemp(prefix="nnInteractiveAgent_"), "segmentation.nrrd")
            sitk.WriteImage(sitkUtils.PullVolumeFromSlicer(labelmap), path, True)
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)
        segments = [{"value": k + 1, "name": seg.GetSegment(i).GetName(), "notes": _get_notes(seg.GetSegment(i))}
                    for k, i in enumerate(ids)]
        self.setStatus(_("Sending {name} to the agent ...").format(name=segmentation.GetName()))
        self._sending_segmentation = True
        self.updateGUI()

        def done(_result, error):
            self._sending_segmentation = False
            try:
                os.remove(path)
                os.rmdir(os.path.dirname(path))
            except OSError:
                pass
            if isinstance(error, EndpointNotFoundError):
                internal = self.internalRadio.checked
                slicer.util.errorDisplay(
                    _("This agent server cannot receive segmentations: its version of nninteractive-agent has no "
                      "segmentation upload (POST /api/objects).\n\n{advice}\n\nEverything else works with this "
                      "server: the agent can still segment the volume you sent.").format(
                        advice=_("Install a version that has it: Configuration tab, nninteractive-agent version, "
                                 "Install recommended version.") if internal else
                        _("Ask whoever runs the server to update nninteractive-agent to a version that has it.")),
                    windowTitle=_("Segmentation upload not available"))
                self.updateGUI()
                return
            if error is not None:
                slicer.util.errorDisplay(_("Could not send the segmentation: {error}").format(error=error))
                self.updateGUI()
                return
            # The sent segments come back as the agent's objects: remove them, so they are not there twice.
            for i in ids:
                seg.RemoveSegment(i)
            self.mirror.set_output("result", segmentation)
            self.updateGUI()

        self.conn.run_async(lambda: self.conn.upload_objects(path, segments, mode), done)

    def onOutputSelected(self, role: str, node) -> None:
        if self._updating_selectors or slicer.mrmlScene.IsClosing() or slicer.mrmlScene.IsBatchProcessing():
            return
        if role == "result" and node is not None:  # a segmentation goes with the volume it was made on
            reference = node.GetNodeReference(node.GetReferenceImageGeometryReferenceRole())
            if reference is not None and reference.IsA("vtkMRMLScalarVolumeNode") \
                    and reference is not self.volumeSelector.currentNode():
                self.volumeSelector.setCurrentNode(reference)
        if not self.outputsAreLive():  # for a volume not sent yet: used once it is
            preferred = self.preferredOutputs()
            other = slicer.mrmlScene.GetNodeByID(preferred.get("work" if role == "result" else "result") or "")
            if node is not None and node is other:
                slicer.util.errorDisplay(_("Segmentation and working segmentation must be two different nodes."))
            elif node is None:
                preferred.pop(role, None)
            else:
                preferred[role] = node.GetID()
            self.updateGUI()
            return
        other = self.mirror.output_node("work" if role == "result" else "result")
        if node is not None and other is not None and node is other:
            slicer.util.errorDisplay(_("Segmentation and working segmentation must be two different nodes."))
            self.updateOutputSelectors()
            return
        self.mirror.set_output(role, node)
        self.mirror.flush()  # creates the new node for "(create new)"
        self._last_flush = time.time()
        self.updateOutputSelectors()

    def onSettingsChanged(self, _index=None) -> None:
        self.conn.send({"cmd": "settings", "model": self.modelCombo.currentData, "effort": self.effortCombo.currentText})

    def onSend(self) -> None:
        text = self.messageEdit.toPlainText().strip()
        if not text or not self.sendButton.enabled:
            return
        if not self.conn.connected:
            self.connectThen(self.onSend)
            return
        if self.image_meta is None:
            self.offerToSendImage()
            return
        if MARKUPS_REFERENCE.search(text):
            volume = self.mirror.volume_node()
            if volume is None:
                slicer.util.errorDisplay(_("The agent's image is not in this scene yet."))
                return
            text, missing, dropped = expand_markups(text, volume)
            if missing:
                slicer.util.errorDisplay(_("No point list, line, curve or ROI is named: {names}").format(
                    names=", ".join(missing)))
                return
            if dropped and not slicer.util.confirmOkCancelDisplay(
                    _("These are outside the agent's image and are left out: {labels}. Send anyway?").format(
                        labels=", ".join(dropped))):
                return
        if not self.confirmDataSharing():
            return
        if self.conn.send({"cmd": "send", "text": text}):
            self.messageEdit.setPlainText("")
            self.chat.verticalScrollBar().value = self.chat.verticalScrollBar().maximum  # follow the answer

    def llmDescription(self) -> tuple:
        """(settings key, description) of where the agent's LLM requests go."""
        name, label = self.agent.get("name", ""), self.agent.get("label", "?")
        model = self.modelCombo.currentText or "?"
        if name.startswith("claude"):
            return "anthropic", _("Anthropic: {label}, model {model}").format(model=model, label=label)
        if name.startswith("codex"):
            return "openai", _("OpenAI: {label}, model {model}").format(model=model, label=label)
        return "server_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", label), _(
            "the model server you configured: {label}. The images leave this computer only if that server runs on "
            "another computer").format(label=label)

    def confirmDataSharing(self) -> bool:
        """Before the first message to an LLM provider: say what the agent sends to it. Can be dismissed for good."""
        provider, description = self.llmDescription()
        key = SETTINGS_PREFIX + "DataSharingAccepted/" + provider
        settings = qt.QSettings()
        if slicer.util.toBool(settings.value(key, False)):
            return True
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.icon = qt.QMessageBox.Warning
        box.windowTitle = _("Data sent to the LLM")
        box.text = _("The agent sends renderings of the image (slices and 3D views) to its LLM: {llm}.").format(
            llm=description)
        box.informativeText = _("Do not use this with patient data unless you are allowed to share it with this "
                                "LLM provider. Click OK to continue.")
        box.detailedText = _("Sent to the LLM: your messages, rendered slices and 3D views of the image and the "
                             "segmentation, the image's size, spacing and intensity statistics, numbers the agent "
                             "computes from the voxels, the volume's name, and the names and notes of the objects. The volume itself is sent to the agent server, "
                             "not to the LLM.")
        box.standardButtons = qt.QMessageBox.Ok | qt.QMessageBox.Cancel
        box.setDefaultButton(qt.QMessageBox.Cancel)
        dontShowAgain = qt.QCheckBox(_("Don't show this again for this LLM provider"))
        box.setCheckBox(dontShowAgain)
        accepted = box.exec_() == qt.QMessageBox.Ok
        box.deleteLater()
        if accepted and dontShowAgain.checked:
            settings.setValue(key, True)
        return accepted

    def offerToSendImage(self) -> None:
        """A message but no image yet: explain, and offer to send the selected volume (the message follows)."""
        text = _("The agent needs an image to work on: send a volume to the agent first.")
        volume = self.volumeSelector.currentNode()
        if volume is None or volume.GetImageData() is None:
            slicer.util.infoDisplay(text, windowTitle=_("No image"))
            return
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.icon = qt.QMessageBox.Question
        box.windowTitle = _("No image")
        box.text = text + "\n\n" + _("Send {name} to the agent? Your message is sent once the image is open.").format(
            name=volume.GetName())
        sendButton = box.addButton(_("Send to agent"), qt.QMessageBox.AcceptRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(sendButton)
        box.exec_()
        if box.clickedButton() == sendButton:
            self._after_image = self.onSend
            self.onUpload()

    def onEnterSendsChanged(self, _checked=None) -> None:
        enter_sends = self.enterSendsCheck.checked
        qt.QSettings().setValue(SETTINGS_PREFIX + "EnterSends", "true" if enter_sends else "false")
        self.sendButton.visible = not enter_sends
        keys = _("Enter sends, Shift+Enter: new line") if enter_sends else _("Ctrl+Enter sends")
        self.messageEdit.placeholderText = _("Describe what to segment, e.g. \"Segment the liver and the spleen, "
                                             "each as its own object.\" ({keys})").format(keys=keys)

    def onEnterKey(self, modifiers: int) -> bool:
        """Enter in the message box: True if it sent the message (else the key inserts a new line)."""
        if self.enterSendsCheck.checked:
            send = not modifiers & int(qt.Qt.ShiftModifier)
        else:
            send = bool(modifiers & int(qt.Qt.ControlModifier))
        if send:
            self.onSend()
        return send

    def onInsertLocation(self) -> None:
        volume = self.mirror.volume_node()
        if volume is None:
            slicer.util.errorDisplay(_("The agent has no image in this scene yet."))
            return
        crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        kji = ras_to_kji(volume, crosshair.GetCrosshairRAS())
        shape = slicer.util.arrayFromVolume(volume).shape
        if any(v < 0 or v >= n for v, n in zip(kji, shape)):
            slicer.util.errorDisplay(_("Place the crosshair inside the image first: hold Shift and move the mouse "
                                       "in a slice view."))
            return
        self.messageEdit.insertPlainText(location_text(kji) + " ")
        self.messageEdit.setFocus()

    def onInsertMenu(self) -> None:
        """Fill the Insert menu: the crosshair position and the markups nodes of the scene."""
        menu = self.insertMenu
        menu.clear()
        action = menu.addAction(_("Crosshair position"))
        action.toolTip = _("Shift + mouse move in a slice view places the crosshair")
        action.connect("triggered()", self.onInsertLocation)
        menu.addSeparator()
        nodes = supported_markups()
        if not nodes:
            menu.addAction(_("(no point lists, lines, curves or ROIs)")).enabled = False
        for node in nodes:
            action = menu.addAction(node.GetName())
            action.connect("triggered()", lambda node_id=node.GetID(): self.onInsertMarkups(node_id))

    def onInsertMarkups(self, node_id: str) -> None:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is not None:
            self.messageEdit.insertPlainText(markups_reference(node) + " ")
            self.messageEdit.setFocus()

    def onChatLink(self, url) -> None:
        link = url.toString()
        if link.startswith("task:") and self.image_meta:
            tasks = self.image_meta.get("tasks") or []
            k = int(link[5:])
            if k < len(tasks):
                self.messageEdit.setPlainText(tasks[k])
        elif link.startswith("img:"):
            self.showImage(link)
        else:
            qt.QDesktopServices.openUrl(url)

    # ---------------------------------------------------------------- inbox
    def poll(self) -> None:
        changed = False
        t_end = time.time() + 0.05  # keep the GUI responsive while a burst arrives
        while time.time() < t_end:
            try:
                item = self.conn.inbox.get_nowait()
            except queue.Empty:
                break
            changed = True
            kind = item[0]
            try:
                if kind == "event":
                    self.onEvent(item[1])
                elif kind == "patch":
                    self.mirror.apply_patch(item[1], item[2])
                elif kind == "call":
                    item[1](item[2], item[3])
                elif kind == "open":
                    self.setStatus(_("Connected"))
                elif kind == "closed":
                    self.setStatus(item[1])
                    if item[2]:
                        self._after_connect = None
                        self.conn.stop()
            except Exception:  # noqa: BLE001 - one bad message must not stop the module
                logging.exception("nnInteractiveAgent: could not process a server message")
        if self.mirror.dirty and time.time() - self._last_flush > 0.3:
            try:
                self.mirror.flush()
            except Exception:  # noqa: BLE001
                logging.exception("nnInteractiveAgent: could not update the segmentation")
            self._last_flush = time.time()
        self.pollInternalServer()
        if self.internal.running != self._server_was_running:
            self._server_was_running = self.internal.running
            changed = True
        if changed:
            self.updateGUI()

    def onEvent(self, ev: dict) -> None:
        t = ev.get("t")
        if t == "hello":
            self.agent = ev.get("agent") or {}
            self.serverInfoLabel.text = _("nnInteractive: {server} (model license: {license})<br>Agent: {agent}").format(
                server=html.escape(str(ev.get("server"))), license=html.escape(str(ev.get("license") or "unknown")),
                agent=html.escape(self.agent.get("label", "?")))
            self.modelCombo.clear()
            for m in self.agent.get("models", []):
                self.modelCombo.addItem(m.get("label", m["id"]), m["id"])
            self.chat.clear()
            self._chat_images = 0
            self.replaying = True
            self.setImage(ev.get("image"), new=False)
        elif t == "replay_done":
            self.replaying = False
            self.chat.verticalScrollBar().value = self.chat.verticalScrollBar().maximum
            self.loadImageIfNeeded()
            action, self._after_connect = self._after_connect, None
            if action is not None:
                self.updateGUI()
                action()
        elif t == "state":
            self.busy, self.ready = bool(ev.get("busy")), bool(ev.get("ready"))
            self.setStatus(ev.get("status", ""))
            self.setSettings(ev.get("model"), ev.get("effort"))
            if self._after_image and self.image_meta is not None and self.pending_upload is None and not self.busy:
                action, self._after_image = self._after_image, None
                qt.QTimer.singleShot(0, action)  # after the buttons are updated for the opened image
        elif t == "image":
            self.addChat(ev)
            if not self.replaying:
                self.setImage(ev.get("image"), new=True)
        elif t == "objects":
            self.mirror.set_objects(ev.get("objects") or [], ev.get("current"), ev.get("current_color"))
        elif t == "focus":
            if self.followButton.checked and self.mirror.volume_node() is not None and ev.get("point"):
                ras = kji_to_ras(self.mirror.volume_node(), ev["point"])
                slicer.modules.markups.logic().JumpSlicesToLocation(*ras, False)
        elif t == "notice":
            self.addChat({"t": "notice", "text": ev.get("text", "")})
            slicer.util.showStatusMessage(ev.get("text", ""), 5000)
        elif t == "mesh":
            pass  # Slicer makes its own surfaces from the segments
        else:
            self.addChat(ev)

    # ---------------------------------------------------------------- image
    def setImage(self, meta, new: bool) -> None:
        """The server's current image; ``new`` when it was just opened (its layers follow as full patches)."""
        if new or meta is None or (self.image_meta or {}).get("n") != meta.get("n") or \
                (self.image_meta or {}).get("key") != meta.get("key"):
            self.mirror.reset()
        self.image_meta = meta
        if meta is None:
            return
        if self.pending_upload is not None and new:
            self.bound_keys[meta["key"]] = self.pending_upload
            self.pending_upload = None
        if not self.replaying:
            self.loadImageIfNeeded()

    def loadImageIfNeeded(self) -> None:
        meta = self.image_meta
        if meta is None or self.mirror.volume_node() is not None:
            return
        tag = f"{meta['key']}:{meta['n']}"
        volume = slicer.mrmlScene.GetNodeByID(self.bound_keys.get(meta["key"]) or "")
        if volume is None:  # loaded before (e.g. before a module reload)?
            volume = next((v for v in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                           if v.GetAttribute(IMAGE_ATTRIBUTE + "Key") == meta["key"]), None)
        if volume is not None:
            self.bindVolume(volume, tag)
            return
        # Tried once per image: this runs again after every reconnect, so a failure must not repeat (or pop up).
        if tag in self._download_failed:
            self.addChat({"t": "notice", "text": self._download_failed[tag]})  # the chat was replayed
            return
        if self._downloading == tag:
            return
        self._downloading = tag
        dest = os.path.join(tempfile.mkdtemp(prefix="nnInteractiveAgent_"), "image.nii.gz")
        self.setStatus(_("Downloading {title} from the server ...").format(title=meta.get("title", "")))

        def done(path, error):
            self._downloading = None
            if self.image_meta is not meta:
                return  # another image was opened meanwhile
            if error is not None:
                text = _("Could not download the image from the server ({error}). Send a volume to the agent, "
                         "or disconnect and connect to try again.").format(error=error)
                self._download_failed[tag] = text
                self.setStatus(text)
                self.addChat({"t": "notice", "text": text})
                self.updateGUI()
                return
            name = re.sub(r"\.(nii\.gz|nii|nrrd|mha|mhd|tiff?|png|jpe?g|npy)$", "", meta.get("title", "image"),
                          flags=re.IGNORECASE)
            volume = slicer.util.loadVolume(path, {"name": name, "show": True})
            volume.SetAttribute(IMAGE_ATTRIBUTE + "Key", meta["key"])
            self.bound_keys[meta["key"]] = volume.GetID()
            self.bindVolume(volume, tag)
            self.updateGUI()

        self.conn.run_async(lambda: self.conn.download(meta["url"], dest), done)

    def bindVolume(self, volume, tag: str) -> None:
        volume.SetAttribute(IMAGE_ATTRIBUTE + "Key", tag.split(":")[0])
        self.mirror.bind(volume, tag)
        preferred = {**self._preferred_outputs.pop("", {}), **self._preferred_outputs.pop(volume.GetID(), {})}
        for role, node_id in preferred.items():  # chosen for this volume before it was sent
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.mirror.set_output(role, node)
        self.volumeSelector.setCurrentNode(volume)
        slicer.util.setSliceViewerLayers(background=volume, fit=True)

    # ------------------------------------------------------------- settings
    def setSettings(self, model, effort) -> None:
        k = self.modelCombo.findData(model)
        if k >= 0:
            self.modelCombo.currentIndex = k
        efforts = next((m.get("efforts", []) for m in (self.agent or {}).get("models", []) if m["id"] == model), [])
        self.effortCombo.clear()
        for e in efforts:
            self.effortCombo.addItem(e)
        k = self.effortCombo.findText(effort or "")
        if k >= 0:
            self.effortCombo.currentIndex = k

    def setStatus(self, text: str) -> None:
        self.statusLabel.text = text

    # ----------------------------------------------------------------- chat
    def addChat(self, ev: dict) -> None:
        t = ev.get("t")
        esc = lambda s: html.escape(str(s or "")).replace("\n", "<br>")  # noqa: E731
        if t == "user":
            self.appendHtml(f"<p><b>You:</b> {esc(ev.get('text'))}</p>")
        elif t == "text":
            self.appendHtml(markdown_html(ev.get("text", "")))
        elif t == "call":
            if ev.get("sub"):
                return
            self.appendHtml(f"<div style='color:gray'>&rarr; {esc(ev.get('tool'))} {esc(_brief_args(ev.get('args')))}</div>")
        elif t == "result":
            if ev.get("error"):
                self.appendHtml(f"<div style='color:#c33'>{esc(ev.get('tool'))}: {esc(ev.get('text', '')[:300])}</div>")
            if self.showImagesCheck.checked and ev.get("image_data"):
                imgs = []
                for data in ev["image_data"]:
                    name = self.addChatImage(data)
                    if name:
                        imgs.append(f"<a href='{name}'><img src='{name}' height='110'></a>")
                if imgs:
                    self.appendHtml("<div>" + " ".join(imgs) + "</div>")
        elif t == "task":
            status = ev.get("status")
            text = ev.get("description") if status == "started" else (ev.get("summary") or "").strip().split("\n")[0]
            self.appendHtml(f"<div style='color:gray'><i>Verifier {esc(status)}: {esc(text)}</i></div>")
        elif t == "done":
            parts = [f"{ev.get('seconds', 0):.0f} s"]
            if ev.get("prompts"):
                parts.append(f"{ev['prompts']} prompts")
            # On a subscription nothing is billed per turn: the cost at API prices would only mislead.
            if ev.get("cost_usd") and (self.agent or {}).get("name") not in SUBSCRIPTION_BACKENDS:
                parts.append(f"${ev['cost_usd']:.2f}")
            label = _("Stopped") if ev.get("text") in ("stopped", "cancelled") else (
                _("Failed") if ev.get("error") else _("Done"))
            self.appendHtml(f"<p style='color:gray'><i>{label} ({', '.join(parts)})</i></p><hr>")
        elif t == "error":
            self.appendHtml(f"<p style='color:#c33'><b>Error:</b> {esc(ev.get('text'))}</p>")
        elif t == "divider":
            self.appendHtml(f"<hr><p style='color:gray'><i>{esc(ev.get('text'))}</i></p>")
        elif t == "notice":
            self.appendHtml(f"<p style='color:#c80'><i>{esc(ev.get('text'))}</i></p>")
        elif t == "image":
            meta = ev.get("image") or {}
            self.appendHtml(f"<hr><p><b>Image:</b> {esc(meta.get('title'))}</p>")
            tasks = meta.get("tasks") or []
            if tasks and not self.replaying:
                links = "<br>".join(f"<a href='task:{k}'>{esc(task)}</a>" for k, task in enumerate(tasks))
                self.appendHtml(f"<p style='color:gray'>Suggested requests:<br>{links}</p>")

    def appendHtml(self, text: str) -> None:
        bar = self.chat.verticalScrollBar()
        at_end = bar.value >= bar.maximum - 4
        cursor = self.chat.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        # A fresh format: a block after an <hr> would otherwise inherit its rule.
        cursor.insertBlock(qt.QTextBlockFormat(), qt.QTextCharFormat())
        cursor.insertHtml(text)
        if at_end:  # follow the output only while at the bottom, not while reading earlier output
            bar.value = bar.maximum

    def addChatImage(self, data):
        if not data:
            return None
        image = qt.QImage()
        if not image.loadFromData(qt.QByteArray(data)):
            return None
        self._chat_images += 1
        name = f"img:{self._chat_images}"
        self.chat.document.addResource(qt.QTextDocument.ImageResource, qt.QUrl(name), image)
        return name

    def showImage(self, name: str) -> None:
        image = self.chat.document.resource(qt.QTextDocument.ImageResource, qt.QUrl(name))
        if image is None:
            return
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(_("What the agent looked at"))
        dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
        label = qt.QLabel()
        label.setPixmap(qt.QPixmap.fromImage(image))
        layout = qt.QVBoxLayout(dialog)
        layout.addWidget(label)
        dialog.show()


def markdown_html(text: str) -> str:
    """The agent's markdown as HTML (Qt's own converter; plain text if it is not available)."""
    try:
        doc = qt.QTextDocument()
        doc.setMarkdown(text)
        return doc.toHtml()
    except Exception:  # noqa: BLE001 - Qt without markdown support
        return "<p>" + html.escape(text).replace("\n", "<br>") + "</p>"


def _brief_args(args) -> str:
    if not isinstance(args, dict):
        return ""
    if "name" in args:
        return str(args["name"])
    if "code" in args:
        return ""
    keys = [k for k in ("axis", "index", "positive") if k in args]
    return ", ".join(f"{k}={args[k]}" for k in keys)


###############################################################################
# Coordinates: server voxel indices (axis0, axis1, axis2) = Slicer (k, j, i)
###############################################################################


def kji_to_ras(volume, kji) -> list:
    """World RAS of a voxel given as [axis0, axis1, axis2] (with the volume's parent transforms)."""
    ijk_to_ras = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(ijk_to_ras)
    ras = ijk_to_ras.MultiplyPoint([float(kji[2]), float(kji[1]), float(kji[0]), 1.0])[:3]
    to_world = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(volume.GetParentTransformNode(), None, to_world)
    return list(to_world.TransformPoint(ras))


def ras_to_kji(volume, ras) -> list:
    """[axis0, axis1, axis2] voxel index (rounded) of a world RAS position."""
    from_world = vtk.vtkGeneralTransform()
    slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volume.GetParentTransformNode(), from_world)
    local = from_world.TransformPoint(ras)
    ras_to_ijk = vtk.vtkMatrix4x4()
    volume.GetRASToIJKMatrix(ras_to_ijk)
    ijk = ras_to_ijk.MultiplyPoint([*local, 1.0])[:3]
    return [int(round(ijk[2])), int(round(ijk[1])), int(round(ijk[0]))]


###############################################################################
# Markups references in messages: [markups NAME] -> voxel coordinates for the agent
###############################################################################

MARKUPS_CLASSES = ("vtkMRMLMarkupsFiducialNode", "vtkMRMLMarkupsLineNode", "vtkMRMLMarkupsCurveNode",
                   "vtkMRMLMarkupsROINode")  # curves include closed curves
MARKUPS_REFERENCE = re.compile(r"\[markups ([^\]]+)\]")


def supported_markups() -> list:
    """Markups nodes a message can refer to, by name."""
    nodes = []
    for cls in MARKUPS_CLASSES:
        nodes += [n for n in slicer.util.getNodesByClass(cls) if n not in nodes]
    return sorted(nodes, key=lambda n: n.GetName().lower())


def markups_reference(node) -> str:
    return f"[markups {node.GetName()}]"


def expand_markups(text: str, volume):
    """``text`` with each [markups NAME] replaced by the voxel coordinates of that node in ``volume``: a
    location per point (point lists, lines, curves) or a voxel box (ROI), taken now. Returns
    (text, names not found, labels of points outside the image, which are left out)."""
    shape = tuple(volume.GetImageData().GetDimensions())[::-1]
    missing, dropped = [], []

    def inside(kji):
        return all(0 <= v < n for v, n in zip(kji, shape))

    def replace(match):
        name = match.group(1).strip()
        node = next((n for n in supported_markups() if n.GetName() == name), None)
        if node is None:
            missing.append(name)
            return match.group(0)
        if node.IsA("vtkMRMLMarkupsROINode"):
            size = node.GetSize()
            object_to_world = node.GetObjectToWorldMatrix()
            corners = [ras_to_kji(volume, object_to_world.MultiplyPoint(
                [sx * size[0] / 2, sy * size[1] / 2, sz * size[2] / 2, 1.0])[:3])
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            lo = [max(0, min(c[a] for c in corners)) for a in range(3)]
            hi = [min(shape[a] - 1, max(c[a] for c in corners)) for a in range(3)]
            if any(lo[a] > hi[a] for a in range(3)):
                dropped.append(name)
                return f"markups {name}: (outside the image)"
            return f"markups {name}: [box axis0={lo[0]}..{hi[0]}, axis1={lo[1]}..{hi[1]}, axis2={lo[2]}..{hi[2]}]"
        parts = []
        for i in range(node.GetNumberOfControlPoints()):
            if node.GetNthControlPointPositionStatus(i) != slicer.vtkMRMLMarkupsNode.PositionDefined:
                continue
            kji = ras_to_kji(volume, node.GetNthControlPointPositionWorld(i))
            label = node.GetNthControlPointLabel(i) or f"{name}-{i + 1}"
            if inside(kji):
                parts.append(f"{label} {location_text(kji)}")
            else:
                dropped.append(label)
        return f"markups {name}: " + (", ".join(parts) or "(no points in the image)")

    return MARKUPS_REFERENCE.sub(replace, text), missing, dropped


###############################################################################
# nnInteractiveAgentLogic
###############################################################################


class nnInteractiveAgentLogic(ScriptedLoadableModuleLogic):
    """The module's work is done by ``AgentConnection`` and ``LabelMirror``; kept for Slicer's conventions."""


###############################################################################
# nnInteractiveAgentTest
###############################################################################


class nnInteractiveAgentTest(ScriptedLoadableModuleTest):
    """Tests that need no server: label patches into segments, coordinates, link parsing."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_parse_server_link()
        self.setUp()
        self.test_mirror()
        self.setUp()
        self.test_coordinates()
        self.setUp()
        self.test_markups()

    def test_parse_server_link(self):
        self.assertEqual(parse_server_link("http://127.0.0.1:8443/?key=abc"), ("http://127.0.0.1:8443/", "abc"))
        self.assertEqual(parse_server_link("gpu:8443"), ("http://gpu:8443/", None))
        self.assertEqual(parse_server_link("https://h/demo/login"), ("https://h/demo/", None))

    def _volume(self, shape):
        volume = slicer.util.addVolumeFromArray(np.zeros(shape, np.int16), name="agent_test")
        volume.SetIJKToRASDirections([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
        volume.SetSpacing(0.8, 1.2, 2.5)
        volume.SetOrigin(10, -20, 30)
        return volume

    def test_mirror(self):
        shape = (6, 7, 8)  # [k, j, i] = server [axis0, axis1, axis2]
        volume = self._volume(shape)
        objects = np.zeros(shape, np.uint8)
        objects[1:3, 2:5, 3:6] = 1
        objects[4, 1, 1] = 2
        mirror = LabelMirror()
        mirror.set_objects([{"id": 1, "name": "liver", "color": "#00e5ff", "voxels": int((objects == 1).sum())},
                            {"id": 2, "name": "spot", "color": "#ff4fd8", "voxels": 1}], None, "#ffd400")
        for name, arr in (("objects", objects), ("current", np.zeros(shape, np.uint8)),
                          ("prompts", np.zeros(shape, np.uint8))):
            header = {"layer": name, "box": [[0, s] for s in shape], "dtype": str(arr.dtype), "shape": list(shape),
                      "full": True, "seq": 1}
            mirror.apply_patch(*decode_patch(encode_patch(header, arr)))
        mirror.bind(volume, "k:1")
        self.assertTrue(mirror.flush())
        result = slicer.mrmlScene.GetNodeByID(mirror.result_id)
        for i in (1, 2):
            got = slicer.util.arrayFromSegmentBinaryLabelmap(result, OBJECT_SEGMENT_PREFIX + str(i), volume)
            np.testing.assert_array_equal(got > 0, objects == i)
        self.assertEqual(result.GetSegmentation().GetSegment(OBJECT_SEGMENT_PREFIX + "1").GetName(), "liver")

        # A partial patch: the agent works on a new object and object 2 is deleted.
        box = [[3, 6], [0, 7], [0, 8]]
        objects[4, 1, 1] = 0
        crop = objects[3:6].copy()
        header = {"layer": "objects", "box": box, "dtype": "uint8", "shape": list(shape), "full": False, "seq": 2}
        mirror.apply_patch(*decode_patch(encode_patch(header, crop)))
        current = np.zeros(shape, np.uint8)
        current[0, 0:2, 0:2] = 1
        header = {"layer": "current", "box": [[0, 1], [0, 2], [0, 2]], "dtype": "uint8", "shape": list(shape),
                  "full": False, "seq": 3}
        mirror.apply_patch(*decode_patch(encode_patch(header, current[0:1, 0:2, 0:2])))
        mirror.set_objects([{"id": 1, "name": "liver", "color": "#00e5ff", "voxels": 18}],
                           {"name": "kidney", "prompts": 1, "voxels": 4}, "#ffd400")
        self.assertTrue(mirror.flush())
        self.assertIsNone(result.GetSegmentation().GetSegment(OBJECT_SEGMENT_PREFIX + "2"))
        work = slicer.mrmlScene.GetNodeByID(mirror.work_id)
        got = slicer.util.arrayFromSegmentBinaryLabelmap(work, CURRENT_SEGMENT, volume)
        np.testing.assert_array_equal(got > 0, current > 0)
        self.assertEqual(work.GetSegmentation().GetSegment(CURRENT_SEGMENT).GetName(), "Working: kidney")

        # Objects into a segmentation selected by the user: it gets the agent's segments and keeps its own;
        # the previous node keeps its segments but is no longer the output.
        mine = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "mine")
        mine.GetSegmentation().AddEmptySegment("own")
        mirror.set_output("result", mine)
        self.assertTrue(mirror.flush())
        self.assertIs(mirror.output_node("result"), mine)
        got = slicer.util.arrayFromSegmentBinaryLabelmap(mine, OBJECT_SEGMENT_PREFIX + "1", volume)
        np.testing.assert_array_equal(got > 0, objects == 1)
        self.assertIsNotNone(mine.GetSegmentation().GetSegment("own"))
        self.assertIsNotNone(result.GetSegmentation().GetSegment(OBJECT_SEGMENT_PREFIX + "1"))
        self.assertIsNone(result.GetAttribute(ROLE_ATTRIBUTE))
        mirror.bind(volume, "k:1")  # a reconnect finds the selected node again
        self.assertTrue(mirror.flush())
        self.assertIs(mirror.output_node("result"), mine)
        mirror.set_output("result", None)  # "(create new)"
        self.assertTrue(mirror.flush())
        self.assertNotIn(mirror.result_id, (result.GetID(), mine.GetID()))
        self.assertIsNotNone(mine.GetSegmentation().GetSegment(OBJECT_SEGMENT_PREFIX + "1"))
        self.delayDisplay("Label mirror test passed")

    def test_coordinates(self):
        volume = self._volume((6, 7, 8))
        kji = [4, 2, 5]
        ras = kji_to_ras(volume, kji)
        self.assertEqual(ras_to_kji(volume, ras), kji)
        self.assertEqual(location_text(kji), "[location axis0=4, axis1=2, axis2=5]")
        self.delayDisplay("Coordinate test passed")

    def test_markups(self):
        volume = self._volume((6, 7, 8))
        points = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "F")
        points.AddControlPoint(kji_to_ras(volume, [2, 3, 4]), "lesion")
        points.AddControlPoint(kji_to_ras(volume, [20, 3, 4]), "far")  # outside the image
        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "box")
        roi.SetCenterWorld(kji_to_ras(volume, [3, 3, 4]))
        roi.SetSize(2.0, 2.0, 2.0)  # +-1 mm: spacing 2.5 (axis0), 1.2 (axis1), 0.8 (axis2)
        self.assertEqual(MARKUPS_REFERENCE.findall("at [markups F] in [markups box]"), ["F", "box"])
        text, missing, dropped = expand_markups("at [markups F] in [markups box], not [markups nope]", volume)
        self.assertEqual(text, "at markups F: lesion [location axis0=2, axis1=3, axis2=4] in markups box: "
                               "[box axis0=3..3, axis1=2..4, axis2=3..5], not [markups nope]")
        self.assertEqual(missing, ["nope"])
        self.assertEqual(dropped, ["far"])
        self.assertEqual([n.GetName() for n in supported_markups()], ["box", "F"])
        self.delayDisplay("Markups test passed")
