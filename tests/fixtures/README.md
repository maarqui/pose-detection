# Test fixtures

Place a small sample image here named `jazz.jpg` (one or two people, from your
concert footage). Then generate the regression baseline once:

```
python -m tests.make_baseline tests/fixtures/jazz.jpg
```

This creates `jazz.keypoints.npy`, which `test_pipeline_regression.py` compares
against. Commit both files so the regression check runs for everyone.
