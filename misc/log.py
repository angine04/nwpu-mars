import os
import re
from datetime import datetime
from inspect import currentframe, getframeinfo


class MeowLogger(object):
    def __init__(self):
        self.logf = None

    def __del__(self):
        if self.logf is not None:
            self.logf.close()

    def __header(self, pid, with_color=True):
        now = datetime.now()
        frameInfo = getframeinfo(currentframe().f_back.f_back)
        if with_color:
            if pid:
                return "[\033[90m{}|\033[0m{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
            return "[\033[90m{}|\033[0m{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)
        else:
            if pid:
                return "[{}|{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
            return "[{}|{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)

    def _strip_ansi(self, text):
        """Remove ANSI escape sequences from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def setLogFile(self, filename):
        if self.logf is not None:
            self.logf.close()
        self.logf = open(filename, "w")

    def log(self, content, muted=False):
        if muted:
            return
        if self.logf is not None:
            # Strip ANSI codes for file output
            clean_content = self._strip_ansi(content)
            self.logf.write(clean_content + "\n")
            self.logf.flush()
            # Also print to console with colors
            print(content)
            return
        print(content)

    def inf(self, line, pid=False, muted=False):
        if self.logf is not None:
            # For file: use header without color
            file_content = self.__header(pid, with_color=False) + line
            # For console: use header with color
            console_content = self.__header(pid, with_color=True) + line
            self.log(console_content, muted)
        else:
            self.log(self.__header(pid, with_color=True) + line, muted)

    def grey(self, line, pid=False, muted=False):
        self.log("{}\033[90m{}\033[0m".format(self.__header(pid), line), muted)

    def red(self, line, pid=False, muted=False):
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)


log = MeowLogger()
