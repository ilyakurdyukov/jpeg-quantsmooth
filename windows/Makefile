
APPNAME ?= jpegqs_gui
SRCNAME ?= jpegqs_gui.c

CFLAGS := -Wall -Wextra -pedantic -O2 -mwindows
CFLAGS += -DWITH_DROP
CFLAGS += -DSHORTCUT_MENU
LIBS := -lcomdlg32

.PHONY: clean all

all: $(APPNAME)

$(APPNAME): dialog.h
dialog.o: dialog.rc dialog.h
	windres $< $@

clean:
	rm -f $(APPNAME) dialog.o

$(APPNAME): $(SRCNAME) dialog.o
	$(CC) $(CFLAGS) -s -o $@ $< dialog.o $(LIBS) -Wl,--gc-sections

