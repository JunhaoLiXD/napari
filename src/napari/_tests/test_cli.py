import gc
import sys
from unittest import mock

import pytest

import napari
from napari import __main__


@pytest.fixture
def mock_run():
    """mock to prevent starting the event loop."""
    with (
        mock.patch('napari.run'),
    ):
        yield napari.run


def test_cli_works(monkeypatch, capsys):
    """Test the cli runs and shows help"""
    monkeypatch.setattr(sys, 'argv', ['napari', '-h'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert 'napari command line viewer.' in str(capsys.readouterr())


def test_cli_shows_plugins(monkeypatch, capsys, tmp_plugin):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--info'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert tmp_plugin.name in str(capsys.readouterr())


def test_cli_parses_unknowns(mock_run, monkeypatch, make_napari_viewer):
    """test that we can parse layer keyword arg variants"""
    mocked_viewer = (
        make_napari_viewer()
    )  # our mock view_path will return this object

    def assert_kwargs(*args, **kwargs):
        assert ['file'] in args
        assert kwargs['contrast_limits'] == (0, 1)

    # testing all the variants of literal_evals
    with mock.patch('napari.__main__.Viewer', return_value=mocked_viewer):
        monkeypatch.setattr(
            napari.components.viewer_model.ViewerModel, 'open', assert_kwargs
        )
        with monkeypatch.context() as m:
            m.setattr(
                sys, 'argv', ['n', 'file', '--contrast-limits', '(0, 1)']
            )
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0,1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0, 1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0,1)'])
            __main__._run()


def test_cli_raises(monkeypatch):
    """test that unknown kwargs raise the correct errors."""
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--nonsense'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: unrecognized argument: --nonsense'

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--gamma'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: argument --gamma expected one argument'


@pytest.mark.usefixtures('builtins')
def test_cli_runscript(monkeypatch, tmp_path, make_napari_viewer):
    """Test that running napari script.py runs a script"""
    v = make_napari_viewer()
    script = tmp_path / 'test.py'
    script.write_text('import napari; v = napari.Viewer(); v.add_points([])')

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', str(script)])
        m.setattr(__main__, 'Viewer', mock.Mock(return_value=v))
        m.setattr(
            'qtpy.QtWidgets.QApplication.exec', lambda *_: None
        )  # revent event loop if run this test standalone
        __main__._run()

    assert len(v.layers) == 1


def test_cli_passes_kwargs(
    mock_run, monkeypatch, make_napari_viewer, mock_qt_method
):
    """test that we can parse layer keyword arg variants"""
    qt_open = mock_qt_method('napari._qt.qt_viewer.QtViewer._qt_open')
    v = make_napari_viewer()

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(sys, 'argv', ['n', 'file', '--name', 'some name'])
        __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_passes_kwargs_stack(
    mock_run, monkeypatch, make_napari_viewer, mock_qt_method
):
    """test that we can parse layer keyword arg variants"""
    qt_open = mock_qt_method('napari._qt.qt_viewer.QtViewer._qt_open')
    v = make_napari_viewer()

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(
            sys,
            'argv',
            [
                'n',
                'file',
                '--stack',
                'file1',
                'file2',
                '--stack',
                'file3',
                'file4',
                '--name',
                'some name',
            ],
        )
        __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[['file1', 'file2'], ['file3', 'file4']],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_initial_fit_uses_shown_canvas_size(qtbot, monkeypatch):
    """Test that CLI file loading fits the image to the shown canvas."""
    import numpy as np

    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from napari.settings import get_settings

    settings = get_settings()
    settings.application.first_time = False
    settings.application.window_size = (1400, 1000)
    settings.application.window_position = (0, 0)

    fit_sizes = []
    open_sizes = []
    window_visible_when_opened = []
    original_reset_view = ViewerModel.reset_view

    def record_fit_size(viewer, *args, **kwargs):
        fit_sizes.append(viewer.canvas.size)
        return original_reset_view(viewer, *args, **kwargs)

    def open_image(qt_viewer, _filenames, **_kwargs):
        open_sizes.append(qt_viewer.viewer.canvas.size)
        window_visible_when_opened.append(
            qt_viewer.viewer.window._qt_window.isVisible()
        )
        qt_viewer.viewer.add_image(np.zeros((200, 2000)))

    def clear_qt_viewer_instances(window):
        window._qt_viewer._instances.clear()

    monkeypatch.setattr(sys, 'argv', ['napari', 'test.tif'])
    monkeypatch.setattr(
        __main__, '_run_configured_startup_script', lambda: None
    )
    monkeypatch.setattr(ViewerModel, 'reset_view', record_fit_size)
    monkeypatch.setattr(QtViewer, '_qt_open', open_image)

    viewer = __main__._build_viewer()
    qtbot.addWidget(
        viewer.window._qt_window,
        before_close_func=clear_qt_viewer_instances,
    )

    assert len(fit_sizes) == 1
    assert open_sizes[0] != (800, 600)
    assert fit_sizes == open_sizes
    assert window_visible_when_opened == [True]


@pytest.mark.parametrize(
    ('open_file', 'welcome_visible'),
    [(False, True), (True, False)],
)
def test_cli_welcome_screen_visibility(
    qtbot, monkeypatch, open_file, welcome_visible
):
    """Test that CLI shows the welcome screen only when no layer was opened."""
    import numpy as np

    from napari._qt.qt_viewer import QtViewer

    welcome_screen_states = []

    def open_image(qt_viewer, filenames, **_kwargs):
        welcome_screen_states.append(qt_viewer.show_welcome_screen)
        if filenames:
            qt_viewer.viewer.add_image(np.zeros((10, 10)))

    def clear_qt_viewer_instances(window):
        window._qt_viewer._instances.clear()

    monkeypatch.setattr(
        sys, 'argv', ['napari'] + (['test.tif'] if open_file else [])
    )
    monkeypatch.setattr(
        __main__, '_run_configured_startup_script', lambda: None
    )
    monkeypatch.setattr(QtViewer, '_qt_open', open_image)

    viewer = __main__._build_viewer()
    qtbot.addWidget(
        viewer.window._qt_window,
        before_close_func=clear_qt_viewer_instances,
    )

    assert welcome_screen_states == [False]
    assert (
        viewer.window._qt_viewer._welcome_widget.isVisible() is welcome_visible
    )


def test_cli_retains_viewer_ref(mock_run, monkeypatch, make_napari_viewer):
    """Test that napari.__main__ is retaining a reference to the viewer."""
    mocked_viewer = (
        make_napari_viewer()
    )  # our mock view_path will return this object
    ref_count = None  # counter that will be updated before __main__._run()

    def _check_refs(**kwargs):
        # when run() is called in napari.__main__, we will call this function
        # it forces garbage collection, and then makes sure that at least one
        # additional reference to our viewer exists.
        gc.collect()
        if sys.getrefcount(mocked_viewer) <= ref_count:  # pragma: no cover
            raise AssertionError(
                'Reference to napari.viewer has been lost by '
                'the time the event loop started in napari.__main__'
            )

    mock_run.side_effect = _check_refs
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file.tif'])
        # return our local v
        with mock.patch(
            'napari.__main__.Viewer', return_value=mocked_viewer
        ) as mock_viewer:
            ref_count = sys.getrefcount(
                mocked_viewer
            )  # count current references
            # mock gui open so we're not opening dialogs/throwing errors on fake path
            with mock.patch(
                'napari._qt.qt_viewer.QtViewer._qt_open', return_value=None
            ) as mock_viewer_open:
                __main__._run()
            mock_viewer.assert_called_once()
            mock_viewer_open.assert_called_once()


def test_cli_plugin_info(monkeypatch):
    """--plugin-info delegates to npe2.cli.list_ and exits."""
    monkeypatch.setattr(sys, 'argv', ['napari', '--plugin-info'])
    with (
        mock.patch('npe2.cli.list_') as mock_list,
        pytest.raises(SystemExit),
    ):
        __main__._run()
    mock_list.assert_called_once()
