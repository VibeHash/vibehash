# vibehash avatar

A small always-on-top pickaxe on the desktop that shows what the worker is doing; click it for the window, right-click for pause, resume and more.

The floating avatar (avatar.py): reads the worker's feed (`worker --watch` or `app`) and moves with it (resting, training, checking, paid), a click opens the window (`window`), a right-click gives the menu, and it can be dragged anywhere. `app --avatar` starts it with the app. Needs a screen and tkinter.

```
vibehash avatar [-h] [--watch PORT]
```

Flags:

- `--watch PORT`: the feed's port (default 7545).
