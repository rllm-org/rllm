import os
import tempfile
import threading
import logging

logger = logging.getLogger(__name__)

_TEXTWORLD_PATCH_LOCK = threading.Lock()
_TEXTWORLD_PATCHED = False 
_ALFWORLD_TMPDIR_LOCK = threading.Lock()
_ALFWORLD_TMPDIR_CONFIGURED = False


def mkdirs(dirpath: str) -> str:
    """ Create a directory and all its parents.

    If the folder already exists, its path is returned without raising any exceptions.

    Arguments:
        dirpath: Path where a folder need to be created.

    Returns:
        Path to the (created) folder.
    """
    try:
        os.makedirs(dirpath)
    except FileExistsError:
        pass

    return dirpath


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Tmpdir Configuration ===
# Force TextWorld/FastDownward temp files to a stable location outside Ray's temp space.


def _default_alfworld_tmpdir() -> str:
    return "/tmp/rllm_tmp"


def _default_alfworld_tmp_target() -> str:
    repo_root = os.path.abspath(os.path.join(_PKG_DIR, "..", "..", ".."))
    return os.path.join(repo_root, "outputs", "alfworld_tmp")


def _ensure_tmpdir_symlink(link_path: str, target_path: str) -> None:
    if os.path.abspath(link_path) == os.path.abspath(target_path):
        os.makedirs(link_path, exist_ok=True)
        return

    os.makedirs(target_path, exist_ok=True)

    if os.path.islink(link_path):
        current_target = os.path.normpath(os.readlink(link_path))
        expected_target = os.path.normpath(target_path)
        if current_target == expected_target:
            return
        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(
            f"ALFWorld tmp link path exists and is not a symlink: {link_path}. "
            f"Remove it or set ALFWORLD_TMPDIR to another short /tmp path."
        )

    os.symlink(target_path, link_path)


def _configure_alfworld_tmpdir() -> str:
    global _ALFWORLD_TMPDIR_CONFIGURED

    with _ALFWORLD_TMPDIR_LOCK:
        tmpdir = (
            os.environ.get("ALFWORLD_TMPDIR")
            or os.environ.get("RLLM_ALFWORLD_TMPDIR")
            or os.environ.get("RLLM_TMPDIR")
            or _default_alfworld_tmpdir()
        )
        tmpdir = os.path.abspath(os.path.expanduser(tmpdir))
        target = (
            os.environ.get("ALFWORLD_TMP_TARGET")
            or os.environ.get("RLLM_ALFWORLD_TMP_TARGET")
            or _default_alfworld_tmp_target()
        )
        target = os.path.abspath(os.path.expanduser(target))

        if tmpdir.startswith("/tmp/"):
            _ensure_tmpdir_symlink(tmpdir, target)
        else:
            os.makedirs(tmpdir, exist_ok=True)

        os.environ["ALFWORLD_TMPDIR"] = tmpdir
        os.environ["ALFWORLD_TMP_TARGET"] = target
        os.environ["TMPDIR"] = tmpdir
        os.environ["TEMP"] = tmpdir
        os.environ["TMP"] = tmpdir
        tempfile.tempdir = tmpdir

        if not _ALFWORLD_TMPDIR_CONFIGURED:
            logger.info("ALFWorld tempfile directory configured: %s -> %s", tmpdir, target)
            _ALFWORLD_TMPDIR_CONFIGURED = True
        return tmpdir



# === TextWorld Parser Patching ===
# TatSu parsers are not thread-safe; wrap them with thread-local instances.


def _patch_textworld_parsers() -> None:
    global _TEXTWORLD_PATCHED

    with _TEXTWORLD_PATCH_LOCK:
        if _TEXTWORLD_PATCHED:
            return
        _TEXTWORLD_PATCHED = True

    _thread_local = threading.local()

    try:
        import textworld.envs.pddl.logic as pddl_logic_mod
        from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics
        from textworld.envs.pddl.logic.parser import PddlLogicParser

        _OrigModelConverter = pddl_logic_mod._ModelConverter

        def _thread_safe_pddl_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "pddl_parser", None)
            if parser is None:
                parser = PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.pddl_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigModelConverter().walk(model)

        pddl_logic_mod._parse_and_convert = _thread_safe_pddl_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch pddl logic parser: %s", e)

    try:
        import textworld.envs.pddl.textgen as textgen_mod
        from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics
        from textworld.envs.pddl.textgen.parser import CSGParser

        _OrigConverter = textgen_mod._Converter

        def _thread_safe_csg_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "csg_parser", None)
            if parser is None:
                parser = CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True)
                _thread_local.csg_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigConverter().walk(model)

        textgen_mod._parse_and_convert = _thread_safe_csg_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch CSG parser: %s", e)

    try:
        import textworld.logic as tw_logic_mod
        from textworld.logic.model import GameLogicModelBuilderSemantics
        from textworld.logic.parser import GameLogicParser

        _OrigLogicConverter = tw_logic_mod._ModelConverter

        def _thread_safe_logic_parse_and_convert(*args, **kwargs):
            parser = getattr(_thread_local, "logic_parser", None)
            if parser is None:
                parser = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
                _thread_local.logic_parser = parser
            model = parser.parse(*args, **kwargs)
            return _OrigLogicConverter().walk(model)

        tw_logic_mod._parse_and_convert = _thread_safe_logic_parse_and_convert
    except Exception as e:
        logger.warning("Failed to patch game logic parser: %s", e)

