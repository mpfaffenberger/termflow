"""Syntax highlighting for code blocks using Pygments.

This module provides syntax highlighting with:
- Language alias mapping (py → python, js → javascript, etc.)
- Lexer caching for performance
- True-color (24-bit) and 256-color terminal support
- Line-by-line and block highlighting
- Automatic language detection
"""

from __future__ import annotations

from pygments import highlight
from pygments.formatters import Terminal256Formatter, TerminalTrueColorFormatter
from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
from pygments.styles import get_all_styles, get_style_by_name
from pygments.util import ClassNotFound

# =============================================================================
# Language Aliases
# =============================================================================

# Map common names/extensions to Pygments lexer names
LANGUAGE_ALIASES: dict[str, str] = {
    # Python
    "py": "python",
    "python3": "python",
    "py3": "python",
    "pyw": "python",
    "pyi": "python",
    "pyx": "cython",
    # JavaScript/TypeScript
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "jsx": "jsx",
    "mjs": "javascript",
    "cjs": "javascript",
    # Ruby
    "rb": "ruby",
    "rake": "ruby",
    "gemspec": "ruby",
    # Shell
    "sh": "bash",
    "shell": "bash",
    "zsh": "zsh",
    "fish": "fish",
    "ps1": "powershell",
    "ps": "powershell",
    "pwsh": "powershell",
    # Systems languages
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "h": "c",
    "hpp": "cpp",
    "hxx": "cpp",
    "rs": "rust",
    "go": "go",
    "golang": "go",
    # Web
    "htm": "html",
    "xhtml": "html",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "styl": "stylus",
    "vue": "vue",
    "svelte": "svelte",
    # Data formats
    "yml": "yaml",
    "jsonc": "json",
    "json5": "json",
    "jsonl": "json",
    "ndjson": "json",
    # Config
    "dockerfile": "docker",
    "containerfile": "docker",
    "makefile": "make",
    "mk": "make",
    "cmake": "cmake",
    "tf": "terraform",
    "hcl": "terraform",
    "tfvars": "terraform",
    # Documentation
    "md": "markdown",
    "markdown": "markdown",
    "rst": "rst",
    "rest": "rst",
    "tex": "latex",
    "latex": "latex",
    # Assembly
    "asm": "nasm",
    "assembly": "nasm",
    "s": "gas",
    # Lisp family
    "lisp": "common-lisp",
    "cl": "common-lisp",
    "scm": "scheme",
    "rkt": "racket",
    "el": "emacs-lisp",
    "elisp": "emacs-lisp",
    "clj": "clojure",
    "cljs": "clojurescript",
    "cljc": "clojure",
    "edn": "clojure",
    # Functional languages
    "erl": "erlang",
    "hrl": "erlang",
    "ex": "elixir",
    "exs": "elixir",
    "heex": "elixir",
    "hs": "haskell",
    "lhs": "haskell",
    "ml": "ocaml",
    "mli": "ocaml",
    "fs": "fsharp",
    "fsi": "fsharp",
    "fsx": "fsharp",
    "f#": "fsharp",
    # JVM languages
    "scala": "scala",
    "sc": "scala",
    "kt": "kotlin",
    "kts": "kotlin",
    "java": "java",
    "groovy": "groovy",
    "gradle": "groovy",
    "gvy": "groovy",
    # Apple ecosystem
    "swift": "swift",
    "m": "objective-c",
    "mm": "objective-c++",
    # Other languages
    "pl": "perl",
    "pm": "perl",
    "perl": "perl",
    "r": "r",
    "R": "r",
    "jl": "julia",
    "lua": "lua",
    "nim": "nim",
    "zig": "zig",
    "v": "v",
    "vlang": "v",
    "d": "d",
    "ada": "ada",
    "adb": "ada",
    "ads": "ada",
    "pas": "pascal",
    "pp": "pascal",
    "vb": "vbnet",
    "vbs": "vbscript",
    "cs": "csharp",
    "c#": "csharp",
    "csx": "csharp",
    "php": "php",
    "php3": "php",
    "php4": "php",
    "php5": "php",
    # Database/Query
    "sql": "sql",
    "mysql": "mysql",
    "pgsql": "postgresql",
    "psql": "postgresql",
    "plpgsql": "plpgsql",
    "sqlite": "sqlite3",
    "sqlite3": "sqlite3",
    # GraphQL/API
    "graphql": "graphql",
    "gql": "graphql",
    "proto": "protobuf",
    "protobuf": "protobuf",
    "thrift": "thrift",
    # Config files
    "toml": "toml",
    "ini": "ini",
    "cfg": "ini",
    "conf": "nginx",
    "nginx": "nginx",
    "apache": "apacheconf",
    "htaccess": "apacheconf",
    # Version control
    "diff": "diff",
    "patch": "diff",
    "git": "diff",
    # Terminal
    "console": "console",
    "terminal": "console",
    "shell-session": "console",
    # Plain text
    "text": "text",
    "txt": "text",
    "plain": "text",
    "": "text",
    "none": "text",
    # Misc
    "xml": "xml",
    "xsl": "xslt",
    "xslt": "xslt",
    "svg": "xml",
    "wasm": "wast",
    "wat": "wast",
    "glsl": "glsl",
    "hlsl": "hlsl",
    "cuda": "cuda",
    "cu": "cuda",
    "opencl": "opencl",
    # Note: "cl" is used for common-lisp above, not opencl
}


class Highlighter:
    """Syntax highlighter using Pygments.

    Provides streaming-friendly syntax highlighting with language detection
    and ANSI true-color output.

    Attributes:
        style_name: Current Pygments style name.
        true_color: Whether to use 24-bit color output.

    Example:
        >>> highlighter = Highlighter()
        >>> # Highlight a complete block
        >>> colored = highlighter.highlight_block(code, "python")
        >>> print(colored)

        >>> # Highlight line by line
        >>> for line in code.splitlines():
        ...     print(highlighter.highlight_line(line, "rust"))
    """

    def __init__(self, style: str = "monokai", true_color: bool = True) -> None:
        """Initialize the highlighter.

        Args:
            style: Pygments style name (monokai, native, vim, dracula, etc.)
            true_color: Use 24-bit true-color (vs 256-color fallback)
        """
        self.style_name = style
        self.true_color = true_color
        self._formatter = self._create_formatter()
        self._lexer_cache: dict[str, object] = {}

    def _create_formatter(self) -> Terminal256Formatter | TerminalTrueColorFormatter:
        """Create the appropriate Pygments formatter."""
        try:
            style = get_style_by_name(self.style_name)
        except ClassNotFound:
            # Fallback to monokai if style not found
            style = get_style_by_name("monokai")

        if self.true_color:
            return TerminalTrueColorFormatter(style=style)
        return Terminal256Formatter(style=style)

    def set_style(self, style: str) -> None:
        """Change the highlighting style.

        Args:
            style: New Pygments style name.

        Example:
            >>> h = Highlighter()
            >>> h.set_style("dracula")
        """
        self.style_name = style
        self._formatter = self._create_formatter()
        # Clear lexer cache since formatter changed
        self._lexer_cache.clear()

    def get_lexer(self, language: str) -> object:
        """Get a Pygments lexer for a language.

        Handles language aliases and caches lexers for performance.

        Args:
            language: Language name (can be an alias like 'py' or 'js')

        Returns:
            Pygments lexer instance.

        Example:
            >>> h = Highlighter()
            >>> lexer = h.get_lexer("py")  # Returns Python lexer
            >>> lexer = h.get_lexer("js")  # Returns JavaScript lexer
        """
        # Normalize and check alias
        lang_lower = language.lower().strip() if language else ""
        canonical = LANGUAGE_ALIASES.get(lang_lower, lang_lower)

        # Check cache
        if canonical in self._lexer_cache:
            return self._lexer_cache[canonical]

        # Try to get lexer
        try:
            lexer = get_lexer_by_name(canonical, stripnl=False)
        except ClassNotFound:
            # Try the original name
            try:
                lexer = get_lexer_by_name(lang_lower, stripnl=False)
            except ClassNotFound:
                # Fallback to plain text
                lexer = TextLexer(stripnl=False)

        self._lexer_cache[canonical] = lexer
        return lexer

    def highlight_line(self, line: str, language: str = "text") -> str:
        """Highlight a single line of code.

        Args:
            line: The code line to highlight.
            language: Language name or alias.

        Returns:
            ANSI-colored string without trailing newline.

        Example:
            >>> h = Highlighter()
            >>> colored = h.highlight_line("print('hello')", "python")
        """
        lexer = self.get_lexer(language)
        # Highlight and strip trailing whitespace/newlines added by Pygments
        result = highlight(line, lexer, self._formatter)
        return result.rstrip("\n\r")

    def highlight_block(self, code: str, language: str = "text") -> str:
        """Highlight a complete code block.

        Args:
            code: Multi-line code string.
            language: Language name or alias.

        Returns:
            ANSI-colored string with newlines preserved.

        Example:
            >>> h = Highlighter()
            >>> code = '''def hello():
            ...     print("Hello!")
            ... '''
            >>> colored = h.highlight_block(code, "python")
        """
        lexer = self.get_lexer(language)
        return highlight(code, lexer, self._formatter)

    def highlight_lines(self, lines: list[str], language: str = "text") -> list[str]:
        """Highlight multiple lines of code.

        More efficient than calling highlight_line() repeatedly as it
        processes all lines together and splits the result.

        Args:
            lines: List of code lines.
            language: Language name or alias.

        Returns:
            List of ANSI-colored strings.
        """
        if not lines:
            return []

        # Join, highlight, then split
        code = "\n".join(lines)
        highlighted = self.highlight_block(code, language)

        # Split back into lines, handling potential trailing newline
        result = highlighted.split("\n")
        # Remove empty trailing element if present
        if result and result[-1] == "":
            result.pop()

        return result

    def guess_and_highlight(self, code: str) -> tuple[str, str]:
        """Guess the language and highlight.

        Uses Pygments' heuristic language detection.

        Args:
            code: Code to analyze and highlight.

        Returns:
            Tuple of (highlighted_code, detected_language_name).

        Example:
            >>> h = Highlighter()
            >>> colored, lang = h.guess_and_highlight("print('hello')")
            >>> print(f"Detected: {lang}")
        """
        try:
            lexer = guess_lexer(code)
            lang_name = lexer.name
        except ClassNotFound:
            lexer = TextLexer()
            lang_name = "Text"

        return highlight(code, lexer, self._formatter), lang_name

    @staticmethod
    def available_styles() -> list[str]:
        """List all available Pygments style names.

        Returns:
            Sorted list of style names.

        Example:
            >>> styles = Highlighter.available_styles()
            >>> 'monokai' in styles
            True
        """
        return sorted(get_all_styles())

    @staticmethod
    def available_languages() -> list[str]:
        """List all available language names and aliases.

        Returns:
            Sorted list of all language identifiers.
        """
        from pygments.lexers import get_all_lexers

        languages: set[str] = set()
        for name, aliases, _, _ in get_all_lexers():
            languages.add(name.lower())
            for alias in aliases:
                languages.add(alias.lower())

        # Add our custom aliases
        languages.update(LANGUAGE_ALIASES.keys())

        return sorted(languages)

    @staticmethod
    def normalize_language(language: str) -> str:
        """Normalize a language name to its canonical form.

        Args:
            language: Language name or alias.

        Returns:
            Canonical language name.

        Example:
            >>> Highlighter.normalize_language("py")
            'python'
            >>> Highlighter.normalize_language("js")
            'javascript'
        """
        lang_lower = language.lower().strip() if language else ""
        return LANGUAGE_ALIASES.get(lang_lower, lang_lower)


# =============================================================================
# Convenience Functions
# =============================================================================

# Module-level highlighter instance for convenience functions
_default_highlighter: Highlighter | None = None


def _get_highlighter() -> Highlighter:
    """Get or create the default highlighter instance."""
    global _default_highlighter
    if _default_highlighter is None:
        _default_highlighter = Highlighter()
    return _default_highlighter


def highlight_code(code: str, language: str = "text", style: str = "monokai") -> str:
    """Highlight code with a one-liner.

    Creates a new Highlighter instance with the specified style.

    Args:
        code: Code to highlight.
        language: Language name or alias.
        style: Pygments style name.

    Returns:
        ANSI-colored string.

    Example:
        >>> colored = highlight_code("print('hello')", "python")
        >>> print(colored)
    """
    h = Highlighter(style=style)
    return h.highlight_block(code, language)


def highlight_line(line: str, language: str = "text") -> str:
    """Highlight a single line using the default highlighter.

    Args:
        line: Code line to highlight.
        language: Language name or alias.

    Returns:
        ANSI-colored string without trailing newline.
    """
    return _get_highlighter().highlight_line(line, language)
