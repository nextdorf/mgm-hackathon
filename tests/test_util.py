from mgm_hackathon.util import LazyVal, _merge_dicts


def test_lazyval_caches_and_invalidates():
  calls = []

  def factory():
    calls.append("called")
    return len(calls)

  lazy = LazyVal(factory)

  assert calls == []
  first_val = lazy.val
  assert first_val == 1
  assert calls == ["called"]

  second_val = lazy.val
  assert second_val == 1
  assert calls == ["called"]

  del lazy.val

  third_val = lazy.val
  assert third_val == 2
  assert calls == ["called", "called"]


def test_merge_dicts_combines_nested_values():
  left = {"file": {"name": "report.pdf"}}
  right = {"file": {"size": 1024}}

  merged = _merge_dicts(left, right)

  assert merged == {"file": {"name": "report.pdf", "size": 1024}}
  # ensure merge is performed in-place for the destination dictionary
  assert right is merged
