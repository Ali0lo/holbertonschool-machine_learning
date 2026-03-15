# Decision Tree Depth

This project adds support for computing the maximum depth of a decision tree.

## Files

- `0-build_decision_tree.py`
  - `Node.max_depth_below(self)`: returns the maximum depth found in the
    subtree rooted at the node.
  - `Leaf.max_depth_below(self)`: returns the leaf depth.
  - `Decision_Tree.depth(self)`: returns the maximum depth of the tree from
    the root.

## Notes

- Root depth is `0`.
- If a node has depth `k`, its children have depth `k + 1`.
- The maximum depth includes leaves.
