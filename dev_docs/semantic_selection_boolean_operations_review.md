# 对《3D 语义选区布尔操作设计计划》的评审意见

> 评审对象:`dev_docs/semantic_selection_boolean_operations.md`
> 评审人:Claude
> 评审基于:`slicer_plugin/SlicerNNInteractive/SlicerNNInteractive.py`、`server/nninteractive_slicer_server/main.py`

## 总体评价

设计文档结构清晰,核心架构判断**基本正确**:布尔运算放在客户端执行、不扩展 server、复用现有的
`get_segment_data()` / `show_segmentation()` / `upload_segment_to_server()` 闭环。阶段拆分和风险清单也务实。

但存在一个关键问题:**文档低估了 Slicer 原生能力**,规划中相当一部分 MVP 是在重复造轮子。
建议在动手前先做减法。

## 1. 关键问题:Slicer 原生已能完成布尔运算

- Segment Editor 自带 **"Logical operators"** 效果,支持 segment 之间的
  `Copy / Add(并) / Subtract(差) / Intersect(交) / Invert / Clear / Fill`。
  文档"阶段 2:segment-to-segment"想手写的 `apply_boolean_operation` 本质上是它的子集。
- Segment Editor 自带 **"Scissors"** 效果,可在 2D/3D 视图中用矩形/自由形状直接切割 segment,
  无需手动做 RAS→IJK 栅格化。文档"阶段 3:ROI box 切割"大部分可被它覆盖。
- 本插件的 `self.ui.editor_widget` 就是一个完整的 `qMRMLSegmentEditorWidget`
  (`SlicerNNInteractive.py:126`),用户**当前已经能**使用上述效果。

**结论**:本次二次开发的真正价值不是"实现布尔运算",而是
(a) 把布尔编辑和 nnInteractive server 同步串起来;
(b) 提供更顺手的语义化 UI。
建议文档明确写出"原生已有什么、我们只补什么",避免重写已有功能并自行承担几何一致性维护成本。

## 2. 关键问题:`@ensure_synched` 已能自动同步本地编辑

文档反复强调"本地改了、server 仍用旧 mask"的状态不一致风险,但现有机制已处理大半:

- `@ensure_synched` 装饰器(`SlicerNNInteractive.py:40`)在**每次 prompt 前**调用
  `selected_segment_changed()`,用 `np.array_equal` 比较当前 segment 与
  `previous_states["segment_data"]`,不一致即自动 `upload_segment_to_server()`。
- 即:用户用任意方式(原生 Logical operators / Scissors / Paint)修改当前 segment 后,
  **下一次 nnInteractive 交互会自动上传新 mask**,无需新增代码。

唯一的不足是同步是"惰性的"——下次 prompt 时才发生。因此真正缺的不是一整套布尔层,
而是一个**"立即同步到 server"的显式入口**,让用户改完即可推送。这能大幅缩减工作量。

## 3. 代码层面的细节与风险补充

- **`upload_segment_to_server()` 不经过 `@ensure_synched`**:阶段 1 示例代码直接调用它没问题
  (布尔操作不改动 volume),但要清楚它不会检查 image 是否变化。
- **server 端 `set_segment` 语义需实测确认**(`main.py:198`):非空 mask 走
  `add_initial_seg_interaction(mask)`,**未先 `reset_interactions()`**;只有空 mask 才 reset。
  文档把上传的 mask 视为"干净的初始分割",需验证 nnInteractive 在已有交互历史上再调用
  `add_initial_seg_interaction` 的行为是否符合预期,并把结论写入文档。
- **`show_segmentation()` 的副作用**:它会写入 `previous_states["segment_data"]`
  (`SlicerNNInteractive.py:984`),因此布尔操作后紧接 `upload_segment_to_server()`
  不会与 `@ensure_synched` 冲突。文档隐含了这一点,但建议显式说明。
- **空结果处理**:Subtract 把 segment 减空时,`show_segmentation` 传入全 0 数组,
  server `set_segment` 会执行 `reset_interactions()`。行为合理,但测试用例需覆盖此情况。
- **坐标顺序**:numpy mask 全程为 `(z, y, x)`;若自行实现 ROI 栅格化路径,务必复用
  `ras_to_xyz`(注意其返回 `(x, y, z)`,需反转)。优先用 Scissors 可直接绕开此坑。

## 4. `Replace` 操作偏冗余

`S' = M` 本质等同于"切换当前选中的 segment",直接在 Segment Editor 中选择另一 segment 即可。
除非需要把 operand 内容复制进 target segment 的 ID,否则不建议作为独立按钮,以减少 UI 噪音。

## 5. 建议修订后的实施优先级

1. 先增加 **"Sync current segment to server"** 按钮(几行代码,直接复用
   `upload_segment_to_server`),立刻打通"原生编辑 + 智能交互"的闭环。
2. 评估原生 Logical operators / Scissors 的覆盖度;若够用,只做**引导式 UI**
   (在插件内暴露入口),不重写算法。
3. 仅当原生 UI 确实笨重时,才做 segment-to-segment 的语义化薄封装
   (复用阶段 1 的 `apply_boolean_operation`)。
4. **临时 operation mask 工作流(原阶段 4)价值最高,可提前**:它把 Paint/Scissors
   变成"一次性操作选区",再 Apply 布尔运算,这是原生没有的体验。
5. ROI 栅格化、旋转 ROI、free-form 3D 选区放在最后。

## 结论

文档方向正确、风险意识到位,但应**先做减法**:先确认 Slicer 原生
Logical operators / Scissors 能覆盖多少,再把工作重心从"实现布尔运算"转向
"server 同步 + 语义化 UI 封装"。最小可用版本可能只需一个显式同步按钮加少量 UI,
而非当前规划的四阶段完整布尔层。

## 待确认事项(交接给 codex / 后续验证)

1. 实测 server `set_segment` 在已有交互历史下接收非空 mask 的行为。
2. 实测原生 Logical operators / Scissors 修改 segment 后,下一次 prompt 是否被
   `@ensure_synched` 正确检测并上传。
3. 确认原生效果在本插件 `editor_widget` 中是否全部可用、UI 是否可接受。
