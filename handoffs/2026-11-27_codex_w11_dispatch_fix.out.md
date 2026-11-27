Cause: exact `preq_bits == sum(glance bits)` failed by 2.62e-10 and 1.75e-10 bits: sequential accumulation differs from Python 3.12’s improved `sum`. Names, hashes and final-part handling were correct. Startup also ignored fetched outputs marked failed.

Fixed [dispatch.py](/root/pdp/kaggle/w11/dispatch.py:73): tightly bounded roundoff handling preserves all other checks; startup validates and accepts both fetched parts without repushing. State needs no manual edit. Offline regression and the 84-part layout check passed.

Measured final-part walls: 333 s and 305 s versus 3,607 s estimated—about 10.8× and 11.8× faster. Split plan and bars unchanged.

Relaunch from `/root/pdp`:

```bash
.venv/bin/python kaggle/w11/dispatch.py --seeds 132 133 134 135 136 137 --cortices bank_1 deep_4
```

No network used; no commit made.