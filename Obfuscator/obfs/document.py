import re

RE_START_WORD = re.compile("[A-Za-z_0-9]*$")
RE_END_WORD = re.compile("^[A-Za-z_0-9]*")


class Document:
    def __init__(
        self,
        source,
        uri=None,
        version=None,
    ):
        self._source = source
        self.uri = uri
        self.version = version

    def __str__(self):
        return str(self.source)

    @property
    def lines(self):
        return self.source.splitlines(True)

    @property
    def source(self):
        return self._source


    def offset_at_position(self, position):
        """Return the byte-offset pointed at by the given position."""
        return position["character"] + len("".join(self.lines[: position["line"]]))

    def word_at_position(self, position):
        """Get the word under the cursor returning the start and end positions."""
        if position["line"] >= len(self.lines):
            return ""

        line = self.lines[position["line"]]
        i = position["character"]
        # Split word in two
        start = line[:i]
        end = line[i:]

        # Take end of start and start of end to find word
        # These are guaranteed to match, even if they match the empty string
        m_start = RE_START_WORD.findall(start)
        m_end = RE_END_WORD.findall(end)

        return m_start[0] + m_end[-1]