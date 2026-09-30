"""Headless live-history tests; no terminal or real agents needed."""

from concurrent.futures import ThreadPoolExecutor
from io import StringIO, UnsupportedOperation

import pytest

from termflow import Parser, Renderer
from termflow.tui import AgentHistory, AgentHistoryViewer, HistoryBuffer, PagerResult


def viewer(history, **kwargs):
    return AgentHistoryViewer(
        history, output=StringIO(), size=lambda: (80, 8), use_alt_screen=False, **kwargs
    )


def test_buffer_chunks_snapshots_and_close():
    buffer = HistoryBuffer()
    assert buffer.write("one\ntw") == 6
    buffer.write("o\n")
    snapshot = buffer.lines()
    snapshot.append("not history")
    assert buffer.getvalue() == "one\ntwo\n"
    assert buffer.lines() == ["one", "two", ""]
    buffer.flush()
    assert not buffer.seekable()
    with pytest.raises(UnsupportedOperation):
        buffer.seek(0)
    with pytest.raises(UnsupportedOperation):
        buffer.truncate(0)
    buffer.close()
    buffer.close()
    assert not buffer.writable()
    assert buffer.getvalue() == "one\ntwo\n"
    with pytest.raises(ValueError):
        buffer.write("late")
    with pytest.raises(TypeError):
        HistoryBuffer().write(b"bytes")


def test_concurrent_writes_are_not_lost():
    buffer = HistoryBuffer()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: buffer.write(f"{n}\n"), range(200)))
    assert sorted(int(line) for line in buffer.lines()[:-1]) == list(range(200))


def test_registration_and_invalid_selection():
    history = AgentHistory()
    with pytest.raises(ValueError):
        viewer(history)
    with pytest.raises(ValueError):
        history.add("")
    main = history.add("main")
    assert history.buffer("main") is main
    with pytest.raises(ValueError):
        history.add("main")
    view = viewer(history)
    with pytest.raises(KeyError):
        view.select("missing")
    assert view.active_agent == "main"
    with pytest.raises(KeyError):
        viewer(history, agent_id="missing")


def test_renderer_output_is_isolated_and_partial_lines_continue():
    history = AgentHistory()
    for name in ("main", "worker"):
        buffer = history.add(name)
        parser = Parser()
        renderer = Renderer(output=buffer, width=70)
        renderer.render_all(parser.parse_line(f"# {name}"))
        renderer.render_all(parser.finalize())
    assert "worker" not in history.buffer("main").getvalue()
    assert "main" not in history.buffer("worker").getvalue()
    view = viewer(history)
    view.select("worker")
    history.buffer("main").write("background ")
    history.buffer("main").write("continued")
    view.select("main")
    assert "background continued" in "\n".join(view._frame())


def test_switch_preserves_scroll_and_follows_hidden_output():
    history = AgentHistory()
    for name in ("main", "worker"):
        history.add(name).write("\n".join(f"{name} {n}" for n in range(20)))
    view = viewer(history)
    assert view.top == 16
    view._handle_key("k")
    assert view.top == 15
    view._handle_key("tab")
    assert view.active_agent == "worker"
    history.buffer("main").write("\nnew main")
    history.buffer("worker").write("\nnew worker")
    view._frame()
    assert view.top == 17
    view._handle_key("left")
    assert view.active_agent == "main"
    assert view.top == 15
    view._handle_key("end")
    view._frame()
    assert view.top == 17
    view._handle_key("home")
    history.buffer("main").write("\nmore")
    view._frame()
    assert view.top == 0
    view.select("main")
    assert view.top == 0


def test_poll_refreshes_live_content_and_resize_without_key():
    history = AgentHistory()
    buffer = history.add("main")
    buffer.write("\n".join(str(n) for n in range(20)))
    terminal = {"size": (80, 8)}
    out = StringIO()
    ticks = iter(["update", "resize", "q"])

    def keys():
        action = next(ticks)
        if action == "update":
            buffer.write("\nlive update")
            return ""
        if action == "resize":
            terminal["size"] = (40, 10)
            return ""
        return action

    view = AgentHistoryViewer(history, output=out, size=lambda: terminal["size"], key_source=keys)
    assert view.run() == PagerResult(key="q")
    assert "live update" in out.getvalue()
    assert out.getvalue().count("\x1b[H") == 4  # entry plus three frames
    assert out.getvalue().count("\x1b[?1049h") == 1
    assert out.getvalue().count("\x1b[?1049l") == 1
    assert view.top == 15


def test_cycle_new_agents_and_custom_handler():
    history = AgentHistory()
    history.add("main")
    view = viewer(history, key_handlers={"x": lambda p: PagerResult(key=p.active_agent)})
    view._handle_key("tab")
    assert view.active_agent == "main"
    history.add("worker")
    view._handle_key("left")
    assert view.active_agent == "worker"
    assert view._handle_key("x") == PagerResult(key="worker")
    view._handle_key("right")
    assert view.active_agent == "main"
    assert view._handle_key("ctrl-c").cancelled


def test_screen_restored_when_key_source_fails():
    history = AgentHistory()
    history.add("main")
    out = StringIO()

    def fail():
        raise RuntimeError("input failed")

    with pytest.raises(RuntimeError):
        AgentHistoryViewer(history, output=out, key_source=fail).run()
    assert out.getvalue().endswith("\x1b[?25h\x1b[?1049l")
