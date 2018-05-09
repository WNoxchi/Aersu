################################################################################
# SECOND VERISON
# Adapted from: https://stackoverflow.com/questions/24072790/detect-key-press-in-python
################################################################################
import curses
import os.linesep

def getKey():
    return curses.wrapper(lambda x: x.getkey())

x = []
while True:
    key = getKey()
    x.append(key)
    # print(key)
    if key == os.linesep:
        break
print(x)

# ORIGINAL: ####################################
# import curses
# import os

# def main(win):
#     win.nodelay(True)
#     key=""
#     win.clear()
#     win.addstr("Detected key:")
#     while 1:
#         try:
#            key = win.getkey()
#            win.clear()
#            win.addstr("Detected key:")
#            win.addstr(str(key))
#            if key == os.linesep:
#               break
#         except Exception as e:
#            # No input
#            pass
#
# curses.wrapper(main)

################################################################################
# FIRST VERISON
# Adapted from: https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
################################################################################
# from sys import stdin, platform
# from tty import setraw
#
# # Platform Check
# if platform[:3] != 'win':
#     from termios import tcgetattr, tcsetattr, TCSADRAIN
#     unix = True
# else:
#     from msvcrt import getch
#     unix = False
#
# def _find_getch(unix=unix):
#     # Non-POSIX. Return msvcrt's (Windows') getch.
#     if not unix:
#         return getch
#     # POSIX system. Create and return a getch that manipulates the tty.
#     def _getch():
#         fd = stdin.fileno()
#         old_settings = tcgetattr(fd)
#         try:
#             setraw(fd)
#             ch = stdin.read(1)
#         finally:
#             tcsetattr(fd, TCSADRAIN, old_settings)
#         return ch
#
#     return _getch()
#
# def getKey():
#     return _find_getch()
#
# # getch = _find_getch()
# # k = getch # if return _getch() otherwise k = getch()
# # print(k)
