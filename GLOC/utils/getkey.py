import curses

def getKey():
    return curses.wrapper(lambda _: _.getkey())
