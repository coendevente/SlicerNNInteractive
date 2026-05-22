# 3D 套索选区（Scissors 画区域 + lasso AI 预览）实现记录

## 背景

`semantic_selection_boolean_operations.md` 目标 #5 提到"扩展更复杂的 3D 自由形状选区"。本功能即落地该扩展，作为 Selection Operations 的第 4 个操作数来源 **index 3 = "Lasso (3D)"**。

用户希望在 3D 视图里用套索框选一个区域，并且**和魔棒完全一致**：先看 AI 预览再决定 Apply。

## 演进

- **v1**（已废弃的做法）：Scissors 画的几何区域**直接**当操作数做布尔运算，没跑 AI，也没有独立预览。
- **v2（当前）**：拆成"输入 / 预览"两层，完全仿魔棒（魔棒是"种子 -> Preview 跑 AI -> 预览 -> Apply 重新跑 AI"）：
  - **输入层**：Scissors 在 3D 视图沿视线挤出画出一个 3D 区域（隐藏节点 `Lasso3dInputSegmentNode (do not touch)`，黄色），对应魔棒"种子"。
  - **预览层**：点 Preview 把输入区域当作 lasso 送进 nnInteractive AI（`/add_lasso_interaction`，带服务器状态备份/恢复），结果写入隐藏节点 `Lasso3dPreviewSegmentNode (do not touch)`（青色），对应魔棒"预览"。预览时隐藏黄色输入避免在 3D 遮挡。
  - **Apply**：重新跑一次 lasso AI 取结果当操作数，对当前 segment 做 Add/Subtract/Intersect，写回并同步 server（和魔棒 Apply 重算一致）。

## 交互流程

选 "Lasso (3D)" -> "Draw lasso (3D)" 在 3D 视图拖闭合环（黄色输入区域，Apply 可用）-> "Preview" 跑 AI 看青色预览（"Clear Preview" 撤销并恢复输入显示）-> 选运算 -> Apply。"Clear Region" 清输入；Undo 沿用 `_record_selection_op_undo`。

## 改动文件（仅客户端，server 不动）

`Resources/UI/SlicerNNInteractive.ui`
- `cbOperandSource` 第 4 项 `Lasso (3D)`。
- `operandLasso3dContainer` 两行四按钮：`pbDrawLasso3d`(checkable) / `pbClearLasso3d`("Clear Region")、`pbPreviewLasso3d` / `pbClearPreviewLasso3d`，外加 `lblLasso3dHint`。

`SlicerNNInteractive.py`
- 实例变量：`_sel_op_lasso3d_input_segment_node`、`_lasso3d_input_segment_id`、`_sel_op_lasso3d_preview_segment_node`、`_lasso3d_preview_segment_id`、`_lasso3d_editor_widget`、`_lasso3d_editor_node`、`_lasso3d_input_observer_tag`、`_lasso3d_in_update`。
- 名称常量：`lasso3d_input_segment_node_name`、`lasso3d_preview_segment_node_name`，均加入 `get_segmentation_node()` 的 `internal_names`。
- 方法：`_get_or_create_lasso3d_segmentation`（输入/预览共用）、`_get_or_create_lasso3d_input_segmentation` / `_get_or_create_lasso3d_preview_segmentation`、`_setup_lasso3d_editor`、`_set_lasso3d_input_visible`、`_activate_lasso3d_scissors`（绑输入节点，观察输入节点）、`_deactivate_lasso3d_scissors`、`_on_lasso3d_input_modified`（刷新 Apply + 重建输入表面）、`on_draw_lasso3d_clicked`、`on_clear_lasso3d_clicked`、`_clear_lasso3d_segment`（输入/预览共用）、`_clear_lasso3d_input_segment` / `_clear_lasso3d_preview_segment`、`_lasso3d_input_to_mask`、`_compute_lasso3d_mask`（仿 `_compute_magic_wand_mask` 的备份/重置/恢复，第 2 步走 `/add_lasso_interaction`）、`_update_lasso3d_preview`、`on_preview_lasso3d_clicked`、`on_clear_preview_lasso3d_clicked`、`_destroy_lasso3d`、`_is_selection_lasso3d_valid`（校验输入非空）。
- 接入：`on_apply_selection_op_clicked` 的 `elif source == 3` 改为 `_compute_lasso3d_mask()` + None 告警，Apply 后清输入+预览；`_refresh_apply_enabled`/`_on_operand_source_changed`/`cleanup`/按钮连线/`on_interaction_node_modified` 均已接好。
- Scissors 参数：`Operation=FillInside`、`Shape=FreeForm`、`SliceCutMode=Unlimited`。

## 设计要点 / 坑

- lasso 提示固定 `positive_click=True`；减除靠布尔 Subtract，不引入极性切换。
- AI 调用必须用魔棒同款服务器状态备份/恢复，避免污染用户当前 nnInteractive 会话。
- 输入区域是实心 3D 块，Preview 后隐藏避免遮挡；Clear Preview 恢复显示。
- `_lasso3d_in_update` 防重入（建闭合表面会再触发 Modified）。
- `_destroy_lasso3d` 移除输入+预览+编辑器三个节点，防重载泄漏。
- 纯客户端，无新增 server endpoint。

## 验证状态

- [x] 静态检查：`.py` 无非 ASCII、`py_compile` 通过、`.ui` 良构 XML；无遗留旧方法名引用。
- [ ] **待在 Slicer 内端到端验证**（需运行中的 server）：
  1. Reload；`operandLasso3dContainer` 有两行四按钮。
  2. 载入体数据、选目标 segment、开体绘制或显示分段表面。
  3. 选 "Lasso (3D)" -> 画闭合环 -> 黄色输入区域、Apply 可用。
  4. "Preview" -> 青色 AI 预览、输入隐藏；"Clear Preview" 撤销并恢复输入。
  5. 选运算 -> Apply -> 目标 segment 变化、提示已同步 server。
  6. Undo 还原；切换来源/进入其它 prompt 时套索停用、节点清理；关闭重载无 `Lasso3dInputSegmentNode`/`Lasso3dPreviewSegmentNode` 残留。

## 给 Codex 的提示

- 把整块 3D 区域当 lasso 送 AI 的分割质量取决于模型；若结果不理想，可考虑限制区域厚度，或改用 scribble 端点（`/add_scribble_interaction`，`_compute_lasso3d_mask` 里把 URL 换掉即可），属调优范畴。
- Scissors 在无头 `qMRMLSegmentEditorWidget` 中的参数常量若与当前 Slicer 版本枚举值不一致，可能"画了但不填充"，优先核对 `SegmentEditorScissorsEffect` 参数取值。
- `_lasso3d_input_to_mask` / `_is_selection_lasso3d_valid` 为纯数组逻辑，便于直接往输入节点写掩膜来单测 Apply（index 3）分支。
