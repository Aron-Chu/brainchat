"""Label Studio: the labeling view (``config``), task JSON (``tasks``), the loopback-only
client (``client``) and the sync between Label Studio and the store (``sync``).

Label Studio runs in Docker on ``127.0.0.1:14610`` (``compose.yaml``). The store stays
the record: Label Studio holds a window's text only while the window does, and every
task it holds is tracked in ``ls_tasks`` so expiry can delete it.
"""
